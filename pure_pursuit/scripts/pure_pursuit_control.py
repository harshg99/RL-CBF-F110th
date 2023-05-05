import numpy as np
import csv
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import argparse

class PurePursuitController:
    def __init__(self, args, waypoints):
        self.args = args
        self.lookahead_distance = args.lookahead_distance
        self.velocity = args.velocity
        self.speed_lookahead_distance = args.speed_lookahead_distance
        self.brake_gain = args.brake_gain
        self.wheel_base = args.wheel_base
        self.visualize = args.visualize
        self.curvature_thresh = args.curvature_thresh
        self.acceleration_lookahead_distance = args.acceleration_lookahead_distance
        self.accel_gain = args.accel_gain
        self.waypoints = waypoints
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