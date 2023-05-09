#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray
import csv
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from copy import deepcopy
import argparse
from pure_pursuit_control import PurePursuitController
import time
import torch
import argparse

from neural_cbf.neural_cbf.datamodules import F110DataModule

from argparse import ArgumentParser
from scripts.create_data import F110System,parse_args

from neural_cbf.neural_cbf.training import NeuralCBFController, F110DynamicsModel,ILController


class NNController:
    def __init__(self, args):

        self.args = args
        if self.args.trt:
            import tensorrt as trt
            import os
            import pycuda.driver as cuda

            self.engine,self.context = self.build_engine(args.onnx_file_path, args)
            if self.engine is None or self.context is None:
                raise SystemExit('ERROR: failed to build the engine or context - check if model exists!')
            for binding in self.engine:
                if self.engine.binding_is_input(binding):  # we expect only one input
                    self.input_shape = self.engine.get_binding_shape(binding)
                    self.input_size = trt.volume(self.input_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
                    self.device_input = cuda.mem_alloc(self.input_size)
                else:  # and one output
                    self.output_shape = self.engine.get_binding_shape(binding)
                    # create page-locked memory buffers (i.e. won't be swapped to disk)
                    self.host_output = cuda.pagelocked_empty(trt.volume(self.output_shape) * self.engine.max_batch_size, dtype=np.float32)
                    self.device_output = cuda.mem_alloc(self.host_output.nbytes)
        else:
            datasource = F110DataModule(args,
                                        model=F110System,
                                        val_split=0.05,
                                        batch_size=50,
                                        quotas={"safe": 0.5, "unsafe": 0.4, "goal": 0.1},
                                        )

            dynamics = F110DynamicsModel(n_dims=5, n_controls=2)
            system = F110System(args)
            if self.args.il:
                dir_path = "neural_cbf/training/checkpoints/" + args.version + "/model.ckpt"
                self.controller = ILController.load_from_checkpoint(dir_path, dynamics_model=dynamics,
                                                                  datamodule=datasource, system=system)
            else:
                dir_path = "neural_cbf/training/checkpoints/" + args.version + "/model.ckpt"
                self.controller = NeuralCBFController.load_from_checkpoint(dir_path, dynamics_model=dynamics,
                                                                             datamodule=datasource, system=system)


    def build_engine(self, onnx_file_path, args):
        import tensorrt as trt
        import os
        import pycuda.driver as cuda

        logger = trt.Logger(trt.Logger.WARNING)
        # initialize TensorRT engine and parse ONNX model
        engine_path = onnx_file_path.split('.')[0] + '16-{}'.format(args.fp16) + '.engine'
        if os.path.exists(engine_path):
            with open(engine_path, 'rb') as f:
                serialized_engine = f.read()
        else:

            builder = trt.Builder(logger)
            explicit_batch = 1 << (int)(trt.NetworconkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(explicit_batch)
            parser = trt.OnnxParser(network, logger)

            # parse model file
            success = parser.parse_from_file(onnx_file_path)
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))

            if not success:
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None, None

            # setting the configuration parameters for the builder
            config = builder.create_builder_config()
            # setting the max workspace size
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            # use FP16 mode if possible
            if args.fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            serialized_engine = builder.build_serialized_network(network, config)
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)

        # generate TensorRT engine optimized for the target platform
        print('Building an engine...')
        engine = trt.Runtime(logger).deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        print("Completed creating Engine")

        return engine, context

    def compute_control_trt(self, state: np.ndarray):
        '''
        state: [x, y, yaw, v, steer]
        Inputs a state and passes it through the NN to get the control
        '''
        # Create a stream in which to copy inputs/outputs and run inference.
        import pycuda.driver as cuda
        stream = cuda.Stream()
        image_path = self.args.img_path + "/" + self.args.img_name
        # preprocess input data
        host_input = np.array(state, dtype=np.float32, order='C')
        cuda.memcpy_htod_async(self.device_input, host_input, stream)

        cuda.memcpy_dtoh_async(self.host_output, self.device_output, stream)
        stream.synchronize()

        # postprocess results
        output_data = torch.Tensor(self.host_output).reshape(list(self.output_shape))
        control  = output_data.cpu().numpy().squeeze()
        return control[0],control[1]

    def compute_control(self,state: np.ndarray):
        control = self.controller.policy_net(torch.tensor(state, dtype=torch.float32))
        control = control.cpu().detach().squeeze().numpy()
        print("shape", control.shape)
        return control[0], control[1]

class SafeNNControl(Node):
    """
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self, args, filename=None):
        super().__init__('safe_nn_control')


        # ROS publishers and subscribers
        self.pub_marker = self.create_publisher(MarkerArray, "marker_array", 10)
        # self.sub_pose = self.create_subscription(PoseStamped, "pf/viz/inferred_pose", self.pose_callback, 10)
        self.sub_pose = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "drive", 10)

        self.controller = NNController(args)
        self.previous_control = np.array([0.0, 0.0])
        self.args = args

        # self.declare_parameter('lookahead_distance', 1.75)
        # self.declare_parameter('velocity', 3.2)
        # self.declare_parameter('speed_lookahead_distance', 2.0)
        # self.declare_parameter('brake_gain', 1.0)
        # self.declare_parameter('wheel_base', 0.33)
        # self.declare_parameter('visualize', False)
        # self.declare_parameter('curvature_thresh', 0.1)
        # self.declare_parameter('acceleration_lookahead_distance', 5.0)
        # self.declare_parameter('accel_gain', 0.0)
        
        # # TODO: create ROS subscribers and publishers
        # self.lookahead_distance = self.get_parameter("lookahead_distance").value
        # self.velocity = self.get_parameter("velocity").value
        # self.speed_lookahead_distance = self.get_parameter("speed_lookahead_distance").value
        # self.brake_gain = self.get_parameter("brake_gain").value
        # self.wheel_base = self.get_parameter("wheel_base").value
        # self.visualize = self.get_parameter("visualize").value
        # self.curvature_thresh = self.get_parameter("curvature_thresh").value
        # self.acceleration_lookahead_distance = self.get_parameter("acceleration_lookahead_distance").value
        # self.accel_gain = self.get_parameter("accel_gain").value

        # self.pub_marker = self.create_publisher(MarkerArray, "marker_array", 10)
        # #self.sub_pose = self.create_subscription(PoseStamped, "pf/viz/inferred_pose", self.pose_callback, 10)
        # self.sub_pose = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, 10)
        # self.pub_drive = self.create_publisher(AckermannDriveStamped, "drive", 10)
        
        # args3['visualize'] = self.visualize
        # args3['curvature_thresh'] = self.curvature_thresh
        # args3['acceleration_lookahead_distance'] = self.acceleration_lookahead_distance
        # args3['accel_gain'] = self.accel_gain
        # args3 = argparse.Namespace(**args)



    # defines the controller for the
    def pose_callback(self, pose_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y

        orientation = [pose_msg.pose.pose.orientation.x,
                       pose_msg.pose.pose.orientation.y,
                       pose_msg.pose.pose.orientation.z,
                       pose_msg.pose.pose.orientation.w]

        # x = pose_msg.pose.position.x
        # y = pose_msg.pose.position.y
        # orientation = [pose_msg.pose.orientation.x,
        #                pose_msg.pose.orientation.y,
        #                pose_msg.pose.orientation.z,
        #                pose_msg.pose.orientation.w]

        euler = R.from_quat(orientation).as_euler('xyz', degrees=False)
        yaw = euler[2]
        state = np.array([x, y, self.previous_control[-1] , self.previous_control[0], yaw]).reshape((1,5))
        velocity, steer = self.controller.compute_control(state)
        # print("type", type(velocity))
        # print("vec", velocity)
        self.previous_control = np.array([velocity, steer])
        drive_msg = AckermannDriveStamped()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = float(velocity)
        drive_msg.drive.steering_angle = float(steer)
        self.pub_drive.publish(drive_msg)
        # TODO: publish drive message, don't forget to limit the steering angle.

def parse_args():
    parser = argparse.ArgumentParser(description='Safe NN Node')
    parser.add_argument('--trt', type=bool, default=False, help='using trt or not')
    parser.add_argument('--il', action='store_true', help='use CBF or IL')
    parser.add_argument('--version', type=str, default='v0')
    parser.add_argument('--onnx_path', type=str, default='data/model.onnx', help='path to onnx model')
    parser.add_argument('--engine_path', type=str, default='data/model.trt', help='path to trt model')
    parser.add_argument('--fp16', action='store_true', help='inference with fp16')
    parser.add_argument('--num_samples', type=int, default=1000000)
    parser.add_argument('--bounds_lower', type=float, default=-14)
    parser.add_argument('--bounds_upper', type=float, default=14)
    parser.add_argument('--steering_max', type=float, default=0.50)
    parser.add_argument('--margin', type=float, default=0.50)
    parser.add_argument('--max_ttc', type=float, default=0.4)
    parser.add_argument('--vel_lower', type=float, default=0.00)
    parser.add_argument('--vel_upper', type=float, default=2.6)
    parser.add_argument('--filename', type=str, default='waypoints.csv') 
    parser.add_argument('--save_dir', type=str, default='trajectory_data/')

    args = parser.parse_args()
    return args

def main(args=None):
    rclpy.init(args=args)
    args2 = parse_args()
    print("PurePursuit Initialized")
    pure_pursuit_node = SafeNNControl(args2)
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
