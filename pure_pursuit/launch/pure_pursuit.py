from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    node1 = Node(
            package='safety_node',
            namespace='',
            executable='safety_node',
            name='safety_node',
            parameters=[
            {"thresh": 0.5},
        ]
        )
    
    node = Node(
        package='pure_pursuit',
        namespace='',
        executable='pure_pursuit_node',
        name='pure_pursuit_node',
        parameters=[
<<<<<<< HEAD
            {"lookahead_distance": 2.25}, #levine 1.25
            {"velocity": 5.5}, #levine 9.0
            {"speed_lookahead_distance": 3.25}, #levine 2.0
            {"brake_gain": 2.4}, #levine 12.0
            {"visualize": True},
            {"wheel_base": 0.45},
            {"acceleration_lookahead_distance": 5.0},
            {"curvature_thresh": 0.1},
            {"accel_gain": 5.0}
=======
            {"lookahead_distance": 2.00}, #levine 1.25
            {"velocity": 5.5}, #levine 9.0 6.0 4.0
            {"speed_lookahead_distance": 3.00}, #levine 2.0
            {"brake_gain": 3.3}, #levine 12.0 
            {"visualize": True},
            {"wheel_base": 0.33},# levine 0.33 0.40
>>>>>>> 35a07aa71e4f645f9ae85d20ab5a8fa7a558f33e
        ]
        )



    ld = LaunchDescription()

    #ld.add_action(node1)
    ld.add_action(node)
    return ld
