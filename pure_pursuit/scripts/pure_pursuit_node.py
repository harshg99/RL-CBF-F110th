#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray
import csv
from scipy.spatial.transform import Rotation as R
# TODO CHECK: include needed ROS msg type headers and libraries

class DataCollection(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('velocity', 3.0)
        self.declare_parameter('speed_lookahead_distance', 1.0)
        self.declare_parameter('brake_gain', 1.0)
        self.declare_parameter('wheel_base', 0.33)
        self.declare_parameter('visualize', false)
        self.declare_parameter('curvature_thresh', 0.1)
        self.declare_parameter('acceleration_lookahead_distance', 5.0)
        self.declare_parameter('accel_gain', 5.0)
        
        # TODO: create ROS subscribers and publishers
        self.lookahead_distance = self.get_parameter("lookahead_distance")
        self.velocity = self.get_parameter("velocity").as_double()
        self.speed_lookahead_distance = self.get_parameter("speed_lookahead_distance")
        self.brake_gain = self.get_parameter("brake_gain")
        self.wheel_base = self.get_parameter("wheel_base")
        self.visualize = self.get_parameter("visualize")
        self.curvature_thresh = self.get_parameter("curvature_thresh")
        self.acceleration_lookahead_distance = self.get_parameter("acceleration_lookahead_distance")
        self.accel_gain = self.get_parameter("accel_gain")

        self.pub_marker = self.create_publisher(MarkerArray, "marker_array", 10)
        self.sub_pose = self.create_subscription(PoseStamped, "pf/viz/inferred_pose", self.pose_callback, 10)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "drive", 10)
        filename = "src/sparse_straights_interpolated.csv"

        #load the csv file data and store the values in a numpy array
        
        positions = []
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                else:
                    positions.append([float(row[0]), float(row[1])])
                    line_count += 1
        
        self.waypoints = np.array(positions)
        

    
   # defines the controller for the      
    def pose_callback(self, pose_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        pos = np.array([[x, y]])
        
        
        closest_idx = np.argmin(np.linalg.norm(self.waypoints - pos, axis=1))
        
        
        #lookahead for goals
        lookahead_points = np.where(np.linalg.norm(self.waypoints - pos, axis=1) < self.lookahead_distance)
        lookahead_points = lookahead_points[lookahead_points > closest_idx]
        lookahead_idx = lookahead_points.max()

        #lookahead for seped reduction
        speed_lookahead_points = np.where(np.linalg.norm(self.waypoints - pos, axis=1) < self.speed_lookahead_distance)
        speed_lookahead_points = speed_lookahead_points[lookahead_points > lookahead_idx]
        spoeed_lookahead_idx = speed_lookahead_points.max()

        #lookahaead for acceleration
        accel_lookahead_points = np.where(np.linalg.norm(self.waypoints - pos, axis=1) < self.accel_lookahead_distance)
        accel_lookahead_points = accel_lookahead_points[lookahead_points > lookahead_idx]
        accel_lookahead_idx = accel_lookahead_points.max()
        
        # COmpute and transform all the ppoitns in the car's frame of reference
        goal_point = self.waypoints[lookahead_idx]
        speed_goal_point = self.waypoints[speed_lookahead_idx]
        accel_goal_point = self.waypoints[accel_lookahead_idx]
        
        orientation_q = [pose_msg.pose.orientation.x, 
                        pose_msg.pose.orientation.y, 
                        pose_msg.pose.orientation.z, 
                        pose_msg.pose.orientation.w]


        orientation_matrix = R.from_quat(orientation_q).inv().as_matrix()
        goal_point_4d = np.concatenate((goal_point, [0, 1]), axis = 0)
        goal_car = np.matmul(orientation_matrix, goal_point_4d)
        speed_goal_point_4d = np.concatenate((speed_goal_point, [0, 1]), axis = 0)
        speed_goal_car = np.matmul(orientation_matrix, speed_goal_point_4d)
        accel_goal_point_4d = np.concatenate((accel_goal_point, [0, 1]), axis = 0)
        accel_goal_car = np.matmul(orientation_matrix, accel_goal_point_4d)

        # compute the drive and velocities
        speed_heading = np.arctan2(speed_goal_car[1], speed_goal_car[0])
        accel_heading = np.arctan2(accel_goal_car[1], accel_goal_car[0])
        brake_amount = self.brake_gain * speed_heading

        lateral_displacement = goal_car[1]
        curvature = (2*lateral_displacement)/np.square(self.lookahead_distance)

        if (abs(curvature) < self.curvature_thresh) and (abs(accel_heading) < 2*self.curvature_thresh):
            velocity += self.accel_gain*abs(2*self.curvature_thresh - abs(accel_heading))
        
        velocity = velocity - brake_amount
        steer = atan(self.wheel_base*curvature)
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = velocity -  brake_amount
        
        pub_drive.publish(drive_msg)
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
