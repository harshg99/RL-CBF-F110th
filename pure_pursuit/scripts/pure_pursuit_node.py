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
# TODO CHECK: include needed ROS msg type headers and libraries

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self, filename = None):
        super().__init__('pure_pursuit_node')
        self.declare_parameter('lookahead_distance', 1.5)
        self.declare_parameter('velocity', 3.0)
        self.declare_parameter('speed_lookahead_distance', 2.0)
        self.declare_parameter('brake_gain', 1.0)
        self.declare_parameter('wheel_base', 0.33)
        self.declare_parameter('visualize', False)
        self.declare_parameter('curvature_thresh', 0.1)
        self.declare_parameter('acceleration_lookahead_distance', 5.0)
        self.declare_parameter('accel_gain', 0.0)
        
        # TODO: create ROS subscribers and publishers
        self.lookahead_distance = self.get_parameter("lookahead_distance").value
        self.velocity = self.get_parameter("velocity").value
        self.speed_lookahead_distance = self.get_parameter("speed_lookahead_distance").value
        self.brake_gain = self.get_parameter("brake_gain").value
        self.wheel_base = self.get_parameter("wheel_base").value
        self.visualize = self.get_parameter("visualize").value
        self.curvature_thresh = self.get_parameter("curvature_thresh").value
        self.acceleration_lookahead_distance = self.get_parameter("acceleration_lookahead_distance").value
        self.accel_gain = self.get_parameter("accel_gain").value

        self.pub_marker = self.create_publisher(MarkerArray, "marker_array", 10)
        #self.sub_pose = self.create_subscription(PoseStamped, "pf/viz/inferred_pose", self.pose_callback, 10)
        self.sub_pose = self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "drive", 10)
        
        if filename is None:
            filename = "/sim_ws/src/pure_pursuit/scripts/raceline_centre_half.csv"
    
        #load the csv file data and store the values in a numpy array
        args = dict()
        args['lookahead_distance'] = self.lookahead_distance
        args['velocity'] = self.velocity
        args['speed_lookahead_distance'] = self.speed_lookahead_distance
        args['brake_gain'] = self.brake_gain
        args['wheel_base'] = self.wheel_base
        args['visualize'] = self.visualize
        args['curvature_thresh'] = self.curvature_thresh
        args['acceleration_lookahead_distance'] = self.acceleration_lookahead_distance
        args['accel_gain'] = self.accel_gain
        args = argparse.Namespace(**args)
        

        positions = []
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            print(csv_reader)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                else:
                    positions.append([float(row[0]), float(row[1])])
                line_count += 1
        
        self.waypoints = np.array(positions)
        print(f'Processed {line_count} lines.')
        self.start_point = self.waypoints[0]
        self.goal_point = self.waypoints[-1]
        self.controller = PurePursuitController(args, self.waypoints.copy())
        


   # defines the controller for the      
    def pose_callback(self, pose_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        orientation = [ pose_msg.pose.pose.orientation.x, 
                        pose_msg.pose.pose.orientation.y, 
                        pose_msg.pose.pose.orientation.z, 
                        pose_msg.pose.pose.orientation.w]
        velocity, steer = self.controller.compute_control(x,y,orientation)
        drive_msg = AckermannDriveStamped()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = steer
        self.pub_drive.publish(drive_msg)
        # TODO: publish drive message, don't forget to limit the steering angle.

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
