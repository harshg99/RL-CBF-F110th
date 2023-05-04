import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray
import csv
from pure_pursuit_node import PurePursuit
from scipy.spatial.transform import Rotation as R
import os
# TODO CHECK: include needed ROS msg type headers and lib

def within_bounds(x,y):

        ## below are the coefficients a, b, and c for all of the outer track lines L and centerlines CL for standard form ax + by = c

        # al1 = -2.2184
        # bl1 = 1
        # cl1 = 38.0722

        # acl1 = -2.2824
        # bcl1 = 1
        # ccl1 = 36.6968

        # al2 = 0.4424
        # bl2 = 1
        # cl2 = 10.6387

        # acl2 = 0.4424
        # bcl2 = 1
        # ccl2 = 9.8021

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
        maxDistance = 0.4

        ## first, check if the point is within the outer track
        withinOutterTrack = (al1*x + bl1*y <= cl1) and (al2*x + bl2*y <= cl2) and (al3*x + bl3*y >= cl3) and (al4*x + bl4*y >= cl4)


        ## now calculate the distances from the point to each of the four centerlines
        d1 = abs(acl3*x + bcl3*y - ccl3) / ((acl3**2 + bcl3**2)**0.5)
        d2 = abs(acl4*x + bcl4*y - ccl4) / ((acl4**2 + bcl4**2)**0.5)
        # d3 = abs(al3*x + bl3*y - cl3) / ((al3**2 + bl3**2)**0.5)
        # d4 = abs(al4*x + bl4*y - cl4) / ((al4**2 + bl4**2)**0.5)


        ## now check if the point is within +- 0.25m of any of the centerlines
        withinCenterlines = (d1 <= maxDistance) or (d2 <= maxDistance) #or (d3 <= maxDistance) or (d4 <= maxDistance)

        return withinOutterTrack and withinCenterlines
    
def ttc_bounds(x,y, yaw, vel):
    ## below are the coefficients a, b, and c for all of the outer track lines L and centerlines CL for standard form ax + by = c

    # al1 = -2.2184
    # bl1 = 1
    # cl1 = 38.0722

    # acl1 = -2.2824
    # bcl1 = 1
    # ccl1 = 36.6968

    # al2 = 0.4424
    # bl2 = 1
    # cl2 = 10.6387

    # acl2 = 0.4424
    # bcl2 = 1
    # ccl2 = 9.8021

    al3 = -2.4726
    bl3 = 1
    cl3 = -24.8661

    ail3 = -2.4726
    bil3 = 1
    cil3 = -18.2661

    al4 = 0.4513
    bl4 = 1
    cl4 = -0.4159

    ail4 = 0.4513
    bil4 = 1
    cil4 = 1.32

    ## below is the max acceptable distance from the centerlines
    maxTTC = 0.3

    # compute the distance to each of the centerlines

    
    ## now calculate the distances from the point to each of the four centerlines
    d1 = (al3*x + bl3*y - cl3 / al3*np.cos(theta) + bl3*np.sin(theta))
    d2 = (al4*x + bl4*y - cl4 / al4*np.cos(theta) + bl4*np.sin(theta))
    d3 = (ail3*x + bil3*y - cil1 / ail3*np.cos(theta) + bil3*np.sin(theta))
    d4 = (ail4*x + bil4*y - cil1 / ail4*np.cos(theta) + bil4*np.sin(theta))

    if d1 < 0:
        d1 = 100
    if d2 < 0:
        d2 = 100
    if d3 < 0:
        d3 = 100
    if d4 < 0:
        d4 = 100

    ttc1 = d1/vel
    ttc2 = d2/vel
    ttc3 = d3/vel
    ttc4 = d4/vel

    # check if any of the ttcs are less than the maxTTC
    withinTTC = (ttc1 <= maxTTC) or (ttc2 <= maxTTC) or (ttc3 <= maxTTC) or (ttc4 <= maxTTC)
    return withinTTC

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--bounds_lower', type=int, default=-13)
    parser.add_argument('--bounds_upper', type=int, default=13)
    parser.add_argument('--vel_lower', type=int, default=0.0)
    parser.add_argument('--vel_upper', type=int, default=4.5)
    parser.add_argument('--filename', type=str, default='waypoints.csv') 
    parser.add_argument('--save_dir', type=str, default='trajectory_data/')
    args = parser.parse_args()
    return args

def generate_data(args):
    
    pure_pursuit_query = PurePursuit(args.filename)
    



    al3 = -2.4726
    bl3 = 1
    cl3 = -24.8661

    ail3 = -2.4726
    bil3 = 1
    cil3 = -18.2661

    al4 = 0.4513
    bl4 = 1
    cl4 = -0.4159

    ail4 = 0.4513
    bil4 = 1
    cil4 = 1.32

    al1 = -2.2184
    bl1 = 1
    cl1 = 38.0722

    al2 = 0.4424
    bl2 = 1
    cl2 = 10.6387


    safe_data = {states: [], controls: []}
    unsafe_data = {states:[], controls: []}
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for i in range(args.num_samples):
        print(i)
        valid = False
        while(not valid):
            x = np.random.uniform(args.bounds_lower, args.bounds_upper)
            y = np.random.uniform(args.bounds_lower, args.bounds_upper)
            withinOutterTrack = (al1*x + bl1*y <= cl1) and (al2*x + bl2*y <= cl2) and (al3*x + bl3*y >= cl3) and (al4*x + bl4*y >= cl4)
            withininnertrack = (ail3*x + bil3*y <= cil3) and (ail4*x + bil4*y <= cil4)
            valid = withinOutterTrack and withininnertrack

        vel = np.random.uniform(args.vel_lower, args.vel_upper)
        yaw = np.random.uniform(-np.pi, np.pi)
        
        orientation = R.from_euler('z', yaw).as_quat()

        
        v_con, theta_con = pure_pursuit_query.compute_control(x,y,orientation)

        safe = within_bounds(x, y) and ttc_bounds(x,y,yaw,vel)

        if trajectory is not None:
            if safe:
                safe_data[states].append([x,y,vel,yaw])
                safe_data[controls].append([v_con, theta_con])
            else:
                unsafe_data[states].append([x,y,vel,yaw])
                unsafe_data[controls].append([0, 0])
    
    np.save(args.save_dir + '/safe_trajectory.npy', safe_data)
    np.save(args.save_dir + '/unsafe_trajectory.npy', unsafe_data)

if __name__ == '__main__':
    args = parse_args()
    generate_data(args)
