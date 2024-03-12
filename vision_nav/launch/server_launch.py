from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from launch.substitutions import Command

def generate_launch_description():
    launch_joystick = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('teleop_twist_joy'),
                'launch',
                'teleop-launch.py'
            ])]),
        launch_arguments = {"config_filepath" : "/home/julian/ros2_ws/src/ros2_lego/charlie/resource/xeox_charlie.config.yaml"}.items())
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("charlie"), "config", "nav.rviz"]
    )
    return LaunchDescription([
        launch_joystick,
        Node(
            package='image_transport',
            executable='republish',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', '/charlie/compressed'),
                ('out', '/image')]
        ),
        Node(
            package='coco_detector',
            executable='coco_detector_node',
            parameters=[{"device": "cuda"}]
       ),
       Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{"robot_description": ParameterValue(Command(['cat /home/julian/ros2_ws/src/ros2_lego/charlie/resource/charlie.urdf']), value_type=str)}]
       ),
       Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=["-d", rviz_config_file]
        )
    ])
