#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from std_msgs.msg import ColorRGBA
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

#import euler_from_quaternion in python
from tf_transformations import euler_from_quaternion
# from tf2_py import transformations


import csv
import argparse
class InteractivePointSelector(Node):

    def __init__(self, filename):
        super().__init__('interactive_point_selector')

        # Publisher for the marker
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)

        # Interactive marker server
        self.server = InteractiveMarkerServer(self, "marker")
        # Subscriber for the 3D point
        self.point_sub = self.create_subscription(PointStamped, 'clicked_point', self.point_callback, qos_profile_sensor_data)

        self.markers_pos = []        
        self.num_marker = 0 

        self.filename = filename
        
    def point_callback(self, msg):
        # Create a new interactive marker for the point
        int_marker = InteractiveMarker()
        int_marker.name = str(self.num_marker)
        int_marker.description = 'Point Marker'
        int_marker.header = msg.header
        int_marker.pose.position = msg.point
        marker = self.create_arrow_marker(msg.point)
        self.markers_pos.append([msg.point.x, msg.point.y, 0.])
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(marker)
        int_marker.controls.append(control)
        
        
        control = InteractiveMarkerControl()
        control.name = "move_x"
        control.orientation_mode = InteractiveMarkerControl.INHERIT
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
   
        int_marker.controls.append(control)
       
        control_y = InteractiveMarkerControl()
        control_y.name = "move_y"
        control_y.orientation.w = 1.
        control_y.orientation.x = 0.
        control_y.orientation.y = 0.
        control_y.orientation.z = 1.
        control_y.always_visible = True
        control_y.orientation_mode = InteractiveMarkerControl.INHERIT
        control_y.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
   
        int_marker.controls.append(control_y)
   
        rot_control = InteractiveMarkerControl()
        rot_control.name = "rotate_z"
        rot_control.orientation.x = 0.
        rot_control.orientation.y = 1.
        rot_control.orientation.z = 0.
        rot_control.orientation.w = 1.
        rot_control.always_visible = True
        rot_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        rot_control.orientation_mode = InteractiveMarkerControl.INHERIT
        # Add the arrow marker to the control and publish it
        
        int_marker.controls.append(rot_control)
        # Publish the interactive marker
        self.server.insert(int_marker, feedback_callback=self.processFeedback)
        self.server.applyChanges()


    
    
    def create_arrow_marker(self, point):
        self.num_marker += 1
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = self.num_marker
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position = point
        q = Quaternion()
        q.x = 0.
        q.y = 0.
        q.z = 0.
        q.w = 1.
        marker.pose.orientation = q
        marker.scale.x = 1.0
        marker.scale.y = 0.2
        marker.scale.z = 0.4
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        marker.lifetime.sec = 10
        return marker
        
    def processFeedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            pose = feedback.pose
            marker_id = feedback.marker_name
            _, _, yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            self.markers_pos[int(marker_id)] = (pose.position.x, pose.position.y, yaw)

    def save_marker_positions(self, filename):
        # Save the marker positions to a CSV file
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'theta'])  # Write header row
            for pos in self.markers_pos:
                writer.writerow(pos) 
    
    def destroy_node(self):
        # Save the marker pos to a file before destroying the node
        self.save_marker_positions(self.filename)
                        
        # Call the parent class's destroy_node() function
        super().destroy_node()
        
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default='waypoints.csv')
    args = parser.parse_args(args=args)
    rclpy.init(args=args.__dict__.values())
    node = InteractivePointSelector(filename=args.filename)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
