#include <sstream>
#include <string>
#include <cmath>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <chrono>
#include <memory>
#include <functional>


//libraries for reading csv
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <Eigen/Dense>
#include<tf2/LinearMath/Quaternion.h>
#include<tf2/LinearMath/Matrix3x3.h>
#include<tf2_geometry_msgs/tf2_geometry_msgs.h>
//import euler_from_quaternion
#include <tf2_eigen/tf2_eigen.h>




/// CHECK: include needed ROS msg type headers and libraries

using namespace std;

using std::placeholders::_1;

using namespace std::chrono_literals;

class PurePursuit : public rclcpp::Node
{
    // Implement PurePursuit
    // This is just a template, you are free to implement your own node!

private:

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_pose;
    // rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose;

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr pub_drive;
    std::vector<std::vector<float>> positions;

    

    rclcpp::Node::SharedPtr node;

    double lookahead_distance = 1.0;
    double velocity = 2.0;
    double speed_lookahead_distance = 1.0;
    double brake_gain = 1.0;
    double wheel_base = 0.33;
    bool visualize = false;
    double curvature_thresh = 0.1;
    double acceleration_lookahead_distance = 5.0;
    double accel_gain = 2.0;
    double pf = false;


public:
    PurePursuit() : Node("pure_pursuit_node")
    {

        
        this->declare_parameter("lookahead_distance", 1.0);
        this->declare_parameter("velocity", 3.0);
        this->declare_parameter("speed_lookahead_distance", 1.0);
        this->declare_parameter("brake_gain", 1.0);
        this->declare_parameter("wheel_base", 0.33);
        this->declare_parameter("visualize", false);
        this->declare_parameter("curvature_thresh", 0.1);
        this->declare_parameter("acceleration_lookahead_distance", 5.0);
        this->declare_parameter("accel_gain", 5.0);
        this->declare_parameter("pf", false); 
        

        this->lookahead_distance = this->get_parameter("lookahead_distance").as_double();
        this->velocity = this->get_parameter("velocity").as_double();
        this->speed_lookahead_distance = this->get_parameter("speed_lookahead_distance").as_double();
        this->brake_gain = this->get_parameter("brake_gain").as_double();
        this->wheel_base = this->get_parameter("wheel_base").as_double();
        this->visualize = this->get_parameter("visualize").as_bool();
        this->curvature_thresh = this->get_parameter("curvature_thresh").as_double();
        this->acceleration_lookahead_distance = this->get_parameter("acceleration_lookahead_distance").as_double();
        this->accel_gain = this->get_parameter("accel_gain").as_double();
        this->pf = this->get_parameter("pf").as_bool();

        // TODO: create ROS subscribers and publishers


        pub_marker = this->create_publisher<visualization_msgs::msg::MarkerArray>("marker_array", 10);
        if(this->pf){
            sub_pose = this->create_subscription<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 100 , std::bind(&PurePursuit::pose_callback, this, _1));
        }else{
            sub_pose = this->create_subscription<nav_msgs::msg::Odometry>("/ego_racecar/odom", 100 , std::bind(&PurePursuit::pose_callback, this, _1));
        }
       
        

        pub_drive = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("drive", 10);

        //read csv and convert to 2d float array
        // std::ifstream file("/home/griffin/Documents/f1tenth_ws/src/lab-5-slam-and-pure-pursuit-team-10/pure_pursuit/src/interpolated_path.csv"); //make sure to place this file
        // std::ifstream file("/home/griffin/Documents/f1tenth_ws/src/f1tenth_gym/sim_ws/src/f1tenth_gym_ros/maps/levine_2nd.csv"); //make sure to place this file

        // std::ifstream file("/home/adithyakvh/Courses/F1-tenth/ROS_Installations_F1_tenth/all_labs_ws/src/lab-5-slam-and-pure-pursuit-team-10/pure_pursuit/src/interpolated_path.csv"); //make sure to place this file
        // std::ifstream file("/home/nvidia/f1tenth_ws/src/lab-5-slam-and-pure-pursuit-team-10/pure_pursuit/src/sparse_straights_interpolated.csv"); //make sure to place this fil
        // std::ifstream file("/home/nvidia/f1tenth_ws/src/lab-5-slam-and-pure-pursuit-team-10/pure_pursuit/src/waypoints_straight_filtered.csv"); //make sure to place this file

	    //std::ifstream file("/home/adithyakvh/Courses/F1-tenth/ROS_Installations_F1_tenth/all_labs_ws/src/lab-5-slam-and-pure-pursuit-team-10/pure_pursuit/src/sparse_straights_interpolated.csv"); //make sure to place this fil
        //std::ifstream file("/home/adithyakvh/Courses/F1-tenth/ROS_Installations_F1_tenth/all_labs_ws/src/lab-5-slam-and-pure-pursuit-team-10/pure_pursuit/src/waypoints_straight_filtered.csv"); //make sure to place this file

        //std::ifstream file("/sim_ws/src/pure_pursuit/src/sparse_straights_interpolated.csv"); //make sure to place this fil
        //std::ifstream file("/sim_ws/src/lab-5-slam-and-pure-pursuit-team-10/pure_pursuit/src/waypoints_straight_filtered.csv"); //make sure to place this file

        std::string line;

        //check if file is empty
        if (!file)
        {
            RCLCPP_INFO(this->get_logger(), "file not found");
        }

        //read file line by line, writing to positions vector
        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::vector<float> pos;

            std::string token;

            //handle case with column headers
            if (line[0] == 'x')
            {
                continue;
            }
            while (std::getline(ss, token, ','))
            {
                pos.push_back(std::stof(token));
            }

            // Add the x y position values to the 2D array
            positions.push_back({pos[0], pos[1]});
        }
    }

    void compute_control(double car_x, double car_y, tf2::Quaternion car_orientation){
        
        
        RCLCPP_INFO(this->get_logger(), "computing control");
        //////////////////////////////////////// WAYPOINT MARKERS ////////////////////////////////////////

        //create the top level marker array
        visualization_msgs::msg::MarkerArray marker_array;
        marker_array.markers.resize(7);

        if (this->visualize){
            // Sphere Marker
            marker_array.markers[0].header.frame_id = "map";
            marker_array.markers[0].id = 0;
            marker_array.markers[0].type = visualization_msgs::msg::Marker::SPHERE;
            marker_array.markers[0].action = visualization_msgs::msg::Marker::MODIFY;
            marker_array.markers[0].pose.position.x = 1.0;
            marker_array.markers[0].pose.position.y = 2.0;
            marker_array.markers[0].pose.position.z = 1.0;
            marker_array.markers[0].scale.x = 1.0;
            marker_array.markers[0].scale.y = 1.0;
            marker_array.markers[0].scale.z = 1.0;
            marker_array.markers[0].color.r = 1.0;
            marker_array.markers[0].color.g = 0.0;
            marker_array.markers[0].color.b = 0.0;
            marker_array.markers[0].color.a = 1.0;

            // CubeList Marker
            marker_array.markers[1].header.frame_id = "map";
            marker_array.markers[1].id = 1;
            marker_array.markers[1].type = visualization_msgs::msg::Marker::CUBE_LIST;
            marker_array.markers[1].action = visualization_msgs::msg::Marker::MODIFY; 
            marker_array.markers[1].scale.x = 0.1;
            marker_array.markers[1].scale.y = 0.1;
            marker_array.markers[1].scale.z = 0.1;
            marker_array.markers[1].color.r = 0.0;
            marker_array.markers[1].color.g = 1.0;
            marker_array.markers[1].color.b = 0.0;
            marker_array.markers[1].color.a = 1.0;
        }


        // Add points to the CubeList marker
        for (std::vector<float> pos : positions) 
        {
            geometry_msgs::msg::Point point;
            point.x = pos[0];
            point.y = pos[1];
            point.z = 0.0;

            marker_array.markers[1].points.push_back(point);
        }


        ///////////////////////////////// WAYPOINT MARKERS END////////////////////////////////////////////
        /////////////////////////////////////////////////// TODO: find the current waypoint to track using methods mentioned in lecture

        //pick the closest point to the car
        int closest_point_index = 0;
        double closest_point_distance = 99799.9;
        for (unsigned int i = 0; i < positions.size(); i++)
        {
            // RCLCPP_INFO(this->get_logger(), "iterating");
            double distance = sqrt(pow(positions[i][0] - car_x, 2) + pow(positions[i][1] - car_y, 2));
            if (distance < closest_point_distance)
            {
                closest_point_distance = distance;
                closest_point_index = i;
                // RCLCPP_INFO(this->get_logger(), "closest point index: %d", closest_point_index);
            }
        }

        if (this->visualize){
            //place a blue marker on this point and add it to the marker array
            marker_array.markers[2].header.frame_id = "map";
            marker_array.markers[2].id = 2;
            marker_array.markers[2].type = visualization_msgs::msg::Marker::CUBE;
            marker_array.markers[2].action = visualization_msgs::msg::Marker::MODIFY;
            marker_array.markers[2].pose.position.x = positions[closest_point_index][0];
            marker_array.markers[2].pose.position.y = positions[closest_point_index][1];
            marker_array.markers[2].pose.position.z = 0.0;
            marker_array.markers[2].scale.x = 0.2;
            marker_array.markers[2].scale.y = 0.2;
            marker_array.markers[2].scale.z = 0.2;
            marker_array.markers[2].color.r = 0.0;
            marker_array.markers[2].color.g = 0.0;
            marker_array.markers[2].color.b = 1.0;
            marker_array.markers[2].color.a = 1.0;
        }

        //now step forward in positions until we find the first point that is at least the lookahead distance away
        unsigned int lookahead_point_index = closest_point_index;
        double lookahead_point_distance = 0.0;
        while (lookahead_point_distance < this->lookahead_distance)
        {

            lookahead_point_index++;
            //wrap around if we reach the end of the array 
            if (lookahead_point_index >= positions.size())
            {
                lookahead_point_index = 0;
            }
            lookahead_point_distance = sqrt(pow(positions[lookahead_point_index][0] - car_x, 2) + pow(positions[lookahead_point_index][1] - car_y, 2));
        }


        //now lets do second speed lookahead point off of the first lookahead point
        unsigned int speed_lookahead_point_index = lookahead_point_index;
        double speed_lookahead_point_distance = 0.0;
        while (speed_lookahead_point_distance < this->speed_lookahead_distance)
        {
            speed_lookahead_point_index++;
            //wrap around if we reach the end of the array
            if (speed_lookahead_point_index >= positions.size())
            {
                speed_lookahead_point_index = 0;
            }
            speed_lookahead_point_distance = sqrt(pow(positions[speed_lookahead_point_index][0] - car_x, 2) + pow(positions[speed_lookahead_point_index][1] - car_y, 2));
        }

        unsigned int accel_lookahead_point_index = speed_lookahead_point_index;
        double accel_lookahead_point_distance = 0.0;
        while (accel_lookahead_point_distance < this->acceleration_lookahead_distance)
        {
            accel_lookahead_point_index++;
            //wrap around if we reach the end of the array
            if (accel_lookahead_point_index >= positions.size())
            {
                accel_lookahead_point_index = 0;
            }
            accel_lookahead_point_distance = sqrt(pow(positions[accel_lookahead_point_index][0] - car_x, 2) + pow(positions[accel_lookahead_point_index][1] - car_y, 2));
        }

        //now lets place a marker on this point

        if (this->visualize){
            marker_array.markers[5].header.frame_id = "map";
            marker_array.markers[5].id = 5;
            marker_array.markers[5].type = visualization_msgs::msg::Marker::CUBE;
            marker_array.markers[5].action = visualization_msgs::msg::Marker::MODIFY;
            marker_array.markers[5].pose.position.x = positions[speed_lookahead_point_index][0];
            marker_array.markers[5].pose.position.y = positions[speed_lookahead_point_index][1];
            marker_array.markers[5].pose.position.z = 0.0;
            marker_array.markers[5].scale.x = 0.2;
            marker_array.markers[5].scale.y = 0.2;
            marker_array.markers[5].scale.z = 0.2;
            marker_array.markers[5].color.r = 1.0;
            marker_array.markers[5].color.g = 0.0;
            marker_array.markers[5].color.b = 0.0;
            marker_array.markers[5].color.a = 1.0;
        }


        // //get the angle between the car and the lookahead point
        // double goalPointX_car = positions[lookahead_point_index][0] - car_x;
        // double goalPointY_car = positions[lookahead_point_index][1] - car_y;


        double goalPointX_map = positions[lookahead_point_index][0];
        double goalPointY_map = positions[lookahead_point_index][1];

        if (this->visualize){
            //place a green marker on this point and add it to the marker array
            marker_array.markers[3].header.frame_id = "map";
            marker_array.markers[3].id = 3;
            marker_array.markers[3].type = visualization_msgs::msg::Marker::CUBE;
            marker_array.markers[3].action = visualization_msgs::msg::Marker::MODIFY;
            marker_array.markers[3].pose.position.x = positions[lookahead_point_index][0];
            marker_array.markers[3].pose.position.y = positions[lookahead_point_index][1];
            marker_array.markers[3].pose.position.z = 0.0;
            marker_array.markers[3].scale.x = 0.2;
            marker_array.markers[3].scale.y = 0.2;
            marker_array.markers[3].scale.z = 0.2;
            marker_array.markers[3].color.r = 0.5;
            marker_array.markers[3].color.g = 0.0;
            marker_array.markers[3].color.b = 0.5;
            marker_array.markers[3].color.a = 1.0;
        }
        /////////////////////////////////////////////////// TODO: transform goal point to vehicle frame of reference

        //get car_yaw from odometry message
        double car_roll = 0.0;
        double car_pitch = 0.0;
        double car_yaw = 0.0;
        tf2::Matrix3x3 m(car_orientation);
        m.getRPY(car_roll, car_pitch, car_yaw);
               // get roll pitch and yaw from quaternion


        //find homogeneous transform from map frame to vehicle frame
        Eigen::Matrix4d T_map_vehicle;
        T_map_vehicle << cos(car_yaw), -sin(car_yaw), 0, car_x,
            sin(car_yaw), cos(car_yaw), 0, car_y,
            0, 0, 1, 0,
            0, 0, 0, 1;

        //find homogeneous transform from vehicle frame to map frame
        Eigen::Matrix4d T_vehicle_map = T_map_vehicle.inverse();

        //find the homogeneous transform from map frame to goal point
        Eigen::Matrix4d T_map_goal;
        T_map_goal << 1, 0, 0, goalPointX_map,
            0, 1, 0, goalPointY_map,
            0, 0, 1, 0,
            0, 0, 0, 1;_
        
        //find the homogeneous transform from vehicle frame to goal point
        Eigen::Matrix4d T_vehicle_goal = T_vehicle_map * T_map_goal;

        //to check, find transformation from map to goal point using car to goal point
        Eigen::Matrix4d T_map_goal_check = T_map_vehicle * T_vehicle_goal;
        
        if (this->visualize){
            //add a yellow marker at this point in the map frame
            marker_array.markers[4].header.frame_id = "map";
            marker_array.markers[4].id = 4;
            marker_array.markers[4].type = visualization_msgs::msg::Marker::CUBE;
            marker_array.markers[4].action = visualization_msgs::msg::Marker::MODIFY;
            marker_array.markers[4].pose.position.x = T_map_goal_check(0, 3);
            marker_array.markers[4].pose.position.y = T_map_goal_check(1, 3);
            marker_array.markers[4].pose.position.z = 1.0;
            marker_array.markers[4].scale.x = 0.2;
            marker_array.markers[4].scale.y = 0.2;
            marker_array.markers[4].scale.z = 0.2;
            marker_array.markers[4].color.r = 1.0;
            marker_array.markers[4].color.g = 1.0;
            marker_array.markers[4].color.b = 0.0;
            marker_array.markers[4].color.a = 1.0;
        }
        
        double speedPointX_map = positions[speed_lookahead_point_index][0];
        double speedPointY_map = positions[speed_lookahead_point_index][1];


        //find homogeneous transform from map frame to speed point
        Eigen::Matrix4d T_map_speed;
        T_map_speed << 1, 0, 0, speedPointX_map,
            0, 1, 0, speedPointY_map,
            0, 0, 1, 0,
            0, 0, 0, 1;

        //find the homogeneous transform from vehicle frame to speed point
        Eigen::Matrix4d T_vehicle_speed = T_vehicle_map * T_map_speed;

        //to check, find transformation from map to speed point using car to speed point
        Eigen::Matrix4d T_map_speed_check = T_map_vehicle * T_vehicle_speed;
_
        //map the magnitude of the speed point heading to range [0, 1]
        double brake_amount = this->brake_gain * abs(speed_heading);

        double accelPointX_map = positions[accel_lookahead_point_index][0];
        double accelPointY_map = positions[accel_lookahead_point_index][1];


        //find homogeneous transform from map frame to speed point
        Eigen::Matrix4d T_map_accel;
        T_map_accel << 1, 0, 0, accelPointX_map,
            0, 1, 0, accelPointY_map,
            0, 0, 1, 0,
            0, 0, 0, 1;

        //find the homogeneous transform from vehicle frame to speed point
        Eigen::Matrix4d T_vehicle_accel = T_vehicle_map * T_map_accel;

        //to check, find transformation from map to speed point using car to speed point
        Eigen::Matrix4d T_map_goal_accel = T_map_vehicle * T_vehicle_accel;


        //find heading of speed point in car frame of reference
        //RCLCPP_INFO(this->get_logger(), "accel heading: %f %F %f", T_vehicle_accel(1, 3), T_vehicle_accel(0, 3), accel_lookahead_point_distance);
        double accel_heading = atan2(T_vehicle_accel(1, 3), T_vehicle_accel(0, 3));


 

        if (this->visualize){
            //add a teal marker of id 6 at this point in the map frame
            marker_array.markers[6].header.frame_id = "map";
            marker_array.markers[6].id = 6;
            marker_array.markers[6].type = visualization_msgs::msg::Marker::CUBE;
            marker_array.markers[6].action = visualization_msgs::msg::Marker::MODIFY;
            marker_array.markers[6].pose.position.x = T_map_speed_check(0, 3);
            marker_array.markers[6].pose.position.y = T_map_speed_check(1, 3);
            marker_array.markers[6].pose.position.z = 1.0;
            marker_array.markers[6].scale.x = 0.2;
            marker_array.markers[6].scale.y = 0.2;
            marker_array.markers[6].scale.z = brake_amount;
            marker_array.markers[6].color.r = 0.0;
            marker_array.markers[6].color.g = 1.0;
            marker_array.markers[6].color.b = 1.0;
            marker_array.markers[6].color.a = 1.0;
        }


        pub_marker->publish(marker_array);


        
        /////////////////////////////////////////////////// TODO: publish drive message, don't forget to limit the steering angle.
        double lateral_displacement = T_vehicle_goal(1, 3);
        double curvature = (2 * lateral_displacement) / pow(this->lookahead_distance, 2);

        double velocity = this->velocity;
        if (abs(curvature) < this->curvature_thresh && abs(accel_heading) < 2 * this->curvature_thresh){
            velocity = velocity + this->accel_gain * abs(2 * this->curvature_thresh - abs(accel_heading));
        }

        RCLCPP_INFO(this->get_logger(), " lookahead: %f, velocity: %f, curvature: %f accel_heading: %f", this->lookahead_distance, velocity, \
        abs(curvature), abs(accel_heading));

        //create a drive message
        double wheel_base = this->wheel_base;
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.frame_id = "base_link";
        drive_msg.drive.steering_angle = atan(curvature * wheel_base);
        drive_msg.drive.speed = velocity - brake_amount;
        //publish the drive message
        pub_drive->publish(drive_msg);
    }

    void PurePursuit::pose_callback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_msg){
                
        double car_x = pose_msg->pose.position.x;
        double car_y = pose_msg->pose.position.y;
        tf2::Quaternion q(
            pose_msg->pose.orientation.x,
            pose_msg->pose.orientation.y,
            pose_msg->pose.orientation.z,
            pose_msg->pose.orientation.w);
        this->compute_control(car_x,car_y,q);
    } //stub code had &pose_msg, the & caused build errors. also said ConstPtr instead of ConstSharedPtr, which also made errors


    void PurePursuit::pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg)
    {
        double car_x = pose_msg->pose.pose.position.x;
        double car_y = pose_msg->pose.pose.position.y;

        tf2::Quaternion q(
            pose_msg->pose.pose.orientation.x,
            pose_msg->pose.pose.orientation.y,
            pose_msg->pose.pose.orientation.z,
            pose_msg->pose.pose.orientation.w);
        
        this->compute_control(car_x,car_y,q);
    }

    ~PurePursuit() {} // destructor, which is called when the object is destroyed,
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    // rclcpp::get_logger("rclcpp").set_level(rclcpp::LoggingSeverity::Debug);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}



// to do:
// make some of these parameters tunable with ros parameters and launch file
// make a launch file
// enable car to go around waypoint loop in either direction, by instead of lookahead_point ++, figure out current car heading ....
