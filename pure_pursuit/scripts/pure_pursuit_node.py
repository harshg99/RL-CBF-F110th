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
        

    def collect_safe_data():
        pass

    def collect_unsafe_data():
        pass

    def within_bounds(x,y):

        ## below are the coefficients a, b, and c for all of the outer track lines L and centerlines CL for standard form ax + by = c

        al1 = -2.2184
        bl1 = 1
        cl1 = 38.0722

        acl1 = -2.2824
        bcl1 = 1
        ccl1 = 36.6968

        al2 = 0.4424
        bl2 = 1
        cl2 = 10.6387

        acl2 = 0.4424
        bcl2 = 1
        ccl2 = 9.8021

        al3 = -2.4726
        bl3 = 1
        cl3 = -24.8661

        acl3 = -2.3683
        bcl3 = 1
        ccl3 = -21.6509

        al4 = 0.4513
        bl4 = 1
        cl4 = -0.4159

        acl4 = 0.4484
        bcl4 = 1
        ccl4 = 0.4599

        ## below is the max acceptable distance from the centerlines
        maxDistance = 0.2

        ## first, check if the point is within the outer track
        withinOutterTrack = (al1*x + bl1*y <= cl1) and (al2*x + bl2*y <= cl2) and (al3*x + bl3*y >= cl3) and (al4*x + bl4*y >= cl4)


        ## now calculate the distances from the point to each of the four centerlines
        d1 = abs(acl1*x + bcl1*y - ccl1) / ((acl1**2 + bcl1**2)**0.5)
        d2 = abs(acl2*x + bcl2*y - ccl2) / ((acl2**2 + bcl2**2)**0.5)
        d3 = abs(acl3*x + bcl3*y - ccl3) / ((acl3**2 + bcl3**2)**0.5)
        d4 = abs(acl4*x + bcl4*y - ccl4) / ((acl4**2 + bcl4**2)**0.5)

        ## now check if the point is within +- 0.25m of any of the centerlines
        withinCenterlines = (d1 <= maxDistance) or (d2 <= maxDistance) or (d3 <= maxDistance) or (d4 <= maxDistance)`

        return withinOutterTrack and withinCenterlines
    
    def ttc_bounds(x,y, yaw, vel):
        ## below are the coefficients a, b, and c for all of the outer track lines L and centerlines CL for standard form ax + by = c

        al1 = -2.2184
        bl1 = 1
        cl1 = 38.0722

        acl1 = -2.2824
        bcl1 = 1
        ccl1 = 36.6968

        al2 = 0.4424
        bl2 = 1
        cl2 = 10.6387

        acl2 = 0.4424
        bcl2 = 1
        ccl2 = 9.8021

        al3 = -2.4726
        bl3 = 1
        cl3 = -24.8661

        acl3 = -2.3683
        bcl3 = 1
        ccl3 = -21.6509

        al4 = 0.4513
        bl4 = 1
        cl4 = -0.4159

        acl4 = 0.4484
        bcl4 = 1
        ccl4 = 0.4599

        ## below is the max acceptable distance from the centerlines
        maxTTC = 0.2

        # compute the distance to each of the centerlines

        
        ## now calculate the distances from the point to each of the four centerlines
        d1 = (acl1*x + bcl1*y - ccl1 / acl1*np.cos(theta) + bcl1*np.sin(theta))
        d2 = (acl2*x + bcl2*y - ccl2 / acl2*np.cos(theta) + bcl2*np.sin(theta))
        d3 = (acl3*x + bcl3*y - ccl1 / acl3*np.cos(theta) + bcl3*np.sin(theta))
        d4 = (acl4*x + bcl4*y - ccl1 / acl4*np.cos(theta) + bcl4*np.sin(theta))

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
