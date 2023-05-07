#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import csv
import os
import numpy as np
from sensor_msgs.msg import LaserScan, PointCloud
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Point32
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import ColorRGBA
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import splprep, splev

x_list, y_list, z_list = [], [], []


def create_straight_segment(pt1, pt2, no_of_points=10):

    unit_vec = (pt2 - pt1) / np.linalg.norm(pt2-pt1)
    segment_length = np.linalg.norm(pt2 - pt1) / no_of_points
    waypoints = pt1 + unit_vec * np.arange(1, no_of_points)


    pts = np.arange(1, no_of_points)
    waypoints = [pt1]
    for each in pts:
        waypoints.append((pt1 + unit_vec * each))
    waypoints.append(pt2)
    waypoints = np.array(waypoints)


# with open("waypoints_1_edit.csv", mode='r') as file:
with open("/home/adithyakvh/Courses/F1-tenth/ROS_Installations_F1_tenth/all_labs_ws/src/lab-5-slam-and-pure-pursuit-team-10/pure_pursuit/src/waypoints_1_sparse.csv", mode='r') as file:

    csvFile = csv.reader(file)
    i = 0
    for i, lines in enumerate(csvFile):
    # for i in range(len(csvFile)):
        if i==0:
            # i+=1    
            continue
        if i in [-1, 6, 7]:
            print("line", lines)
            create_straight_segment(lines, next(csvFile))
        # i += 1
        # print("line", lines)
        x = float(lines[0])
        y = float(lines[1])
        print("x", x)

        x_list.append(x)
        y_list.append(y)

exit()
# pt 6 and pt 7  8.83 and -7.12
# pt last and pt 1
print(x_list)
print(y_list)        
plt.figure()
plt.plot(x_list, y_list, 'bo')
plt.show()
        
tck, u = splprep([x_list, y_list], s=1, per=True)
# new_points = splev(u, tck)
new_points = splev(np.linspace(0, 1, 100), tck)

print("new_points", new_points)


file = open("waypoints_interpolated.csv", 'w', newline='')
with file:
    write = csv.writer(file)
    write.writerow(['x', 'y', 'z'])  # Write header row
    for x_p, y_p in zip(new_points[0], new_points[1]):
        write.writerow((x_p, y_p, 0)) 


plt.figure()
plt.plot(new_points[0], new_points[1])
plt.show()