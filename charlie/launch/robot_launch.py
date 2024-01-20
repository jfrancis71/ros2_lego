from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_brickpi3_charlie = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('brickpi3_charlie'),
                'launch',
                'brickpi3_motors_launch.py'
            ])
        ]))
    return LaunchDescription([
        launch_brickpi3_charlie,
        Node(
            package='brickpi3_sensors',
            executable='compass_node',
        ),
        Node(
            package='brickpi3_sensors',
            executable='ultrasonic_distance_node',
            parameters=[{"lego_port": "PORT_2"}]
        ),
        Node(
            package='image_tools',
            executable='cam2image',
            remappings=[
                ('/image', '/charlie/image')]
        ),
        Node(
            package='image_transport',
            executable='republish',
            arguments=['raw', 'compressed'],
            remappings=[
                ('in', '/charlie/image'),
                ('out/compressed', '/charlie/compressed')]
        ),
    ])
