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
# TODO CHECK: include needed ROS msg type headers and libraries

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self, filename = None):
        super().__init__('pure_pursuit_node')
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('velocity', 3.6)
        self.declare_parameter('speed_lookahead_distance', 1.2)
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
            filename = "/sim_ws/src/pure_pursuit/scripts/raceline_centre.csv"
    
        #load the csv file data and store the values in a numpy array
        
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
        

    def compute_control(self, x, y,orientation):
        pos = np.array([[x, y]])
        
        
        closest_idx = np.argmin(np.linalg.norm(self.waypoints - pos, axis=1))
        
        
        #lookahead for goals
        #print(self.lookahead_distance)

        if closest_idx == self.waypoints.shape[0] - 1:
            closest_idx = 0
        
        lookahead_points = np.where(np.linalg.norm(self.waypoints - pos, axis=1) < self.lookahead_distance)[0]
        #print(lookahead_points)
        # check for wrap around
        if lookahead_points.max() == self.waypoints.shape[0] - 1 \
            and lookahead_points.min() == 0:
            lookahead_points_inv = np.where(np.linalg.norm(self.waypoints - pos, axis=1) > self.lookahead_distance)[0]
            lookahead_idx = lookahead_points_inv.min()
        else:
            lookahead_points = lookahead_points[np.where(lookahead_points > closest_idx)[0]]
            lookahead_idx = lookahead_points.max()

        if lookahead_idx == self.waypoints.shape[0] - 1:
            lookahead_idx = 0
        #lookahead for seped reduction
        speed_lookahead_points = np.where(np.linalg.norm(self.waypoints - pos, axis=1) < self.speed_lookahead_distance)[0]
        if speed_lookahead_points.max() == self.waypoints.shape[0] - 1 \
            and speed_lookahead_points.min() == 0:
            speed_lookahead_points_inv = np.where(np.linalg.norm(self.waypoints - pos, axis=1) > self.speed_lookahead_distance)[0]
            speed_lookahead_idx = speed_lookahead_points_inv.min()
        else:
            speed_lookahead_points = speed_lookahead_points[np.where(speed_lookahead_points > lookahead_idx)[0]]
            speed_lookahead_idx = speed_lookahead_points.max()
        
        if speed_lookahead_idx == self.waypoints.shape[0] - 1:
            speed_lookahead_idx = 0
        #lookahaead for acceleration
        accel_lookahead_points = np.where(np.linalg.norm(self.waypoints - pos, axis=1) < self.acceleration_lookahead_distance)[0]
        if accel_lookahead_points.max() == self.waypoints.shape[0] - 1 \
            and accel_lookahead_points.min() == 0:
            accel_lookahead_points_inv = np.where(np.linalg.norm(self.waypoints - pos, axis=1) > self.acceleration_lookahead_distance)[0]
            accel_lookahead_idx = accel_lookahead_points_inv.min()
        else:
            accel_lookahead_points = accel_lookahead_points[np.where(accel_lookahead_points > speed_lookahead_idx)[0]]
            accel_lookahead_idx = accel_lookahead_points.max()
        if accel_lookahead_idx == self.waypoints.shape[0] - 1:
            accel_lookahead_idx = 0

        # COmpute and transform all the ppoitns in the car's frame of reference
        goal_point = self.waypoints[lookahead_idx]
        speed_goal_point = self.waypoints[speed_lookahead_idx]
        accel_goal_point = self.waypoints[accel_lookahead_idx]
        
        orientation_q = deepcopy(orientation)

        orientation_matrix = R.from_quat(orientation_q).as_matrix()

        so3_matrix = np.eye(4)
        so3_matrix[0:3, 0:3] = orientation_matrix
        so3_matrix[0:3, 3] = [x, y, 0]
        so3_matrix = np.linalg.inv(so3_matrix)

        goal_point_4d = np.concatenate((goal_point, [0, 1]), axis = 0)
        goal_car = np.matmul(so3_matrix, goal_point_4d)

        speed_goal_point_4d = np.concatenate((speed_goal_point, [0, 1]), axis = 0)
        speed_goal_car = np.matmul(so3_matrix, speed_goal_point_4d)

        accel_goal_point_4d = np.concatenate((accel_goal_point, [0, 1]), axis = 0)
        accel_goal_car = np.matmul(so3_matrix, accel_goal_point_4d)

        # compute the drive and velocities
        speed_heading = np.arctan2(speed_goal_car[1], speed_goal_car[0])
        accel_heading = np.arctan2(accel_goal_car[1], accel_goal_car[0])
        brake_amount = self.brake_gain * speed_heading

        lateral_displacement = goal_car[1]
        curvature = (2*lateral_displacement)/np.square(self.lookahead_distance)
        
        velocity = self.velocity - brake_amount
        if (abs(curvature) < self.curvature_thresh) and (abs(accel_heading) < 2*self.curvature_thresh):
            velocity += self.accel_gain*abs(2*self.curvature_thresh - abs(accel_heading))
        
       
        steer = np.arctan(self.wheel_base*curvature)
        return velocity, steer

   

   # defines the controller for the      
    def pose_callback(self, pose_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        x = pose_msg.pose.pose.position.x
        y = pose_msg.pose.pose.position.y
        orientation = [ pose_msg.pose.pose.orientation.x, 
                        pose_msg.pose.pose.orientation.y, 
                        pose_msg.pose.pose.orientation.z, 
                        pose_msg.pose.pose.orientation.w]
        velocity, steer = self.compute_control(x,y,orientation)
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
