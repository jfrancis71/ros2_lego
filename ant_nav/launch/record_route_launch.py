from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from launch.substitutions import Command
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    route_folder_launch_arg = DeclareLaunchArgument(
      'route_folder')

    launch_joystick = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('teleop_twist_joy'),
                'launch',
                'teleop-launch.py'
            ])]),
        launch_arguments = {"config_filepath" : "/home/julian/ros2_humble/src/ros2_lego/charlie/resource/xeox_charlie.config.yaml"}.items())

    decompress_image_node = Node(
            package='image_transport',
            executable='republish',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', '/thomas/compressed'),
                ('out', '/image')])

    record_route_node = Node(
        package='ant_nav',
        executable='record_route_node',
        parameters=[{
            'route_folder': LaunchConfiguration('route_folder')}]
        )

    return LaunchDescription([
        route_folder_launch_arg,
        launch_joystick,
        decompress_image_node,
        record_route_node
    ])

