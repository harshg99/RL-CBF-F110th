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
                dir_path = "neural_cbf/training/checkpoints/" + args.version + "/model-v5.ckpt"
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
        control = self.controller.policy_net(state)
        control = control.cpu().detach().squeeze().numpy()
        return control

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
        self.previous_control = np.array([velocity, steer])
        drive_msg = AckermannDriveStamped()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = steer
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
    args = parser.parse_args()
    return args

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = SafeNNControl()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
