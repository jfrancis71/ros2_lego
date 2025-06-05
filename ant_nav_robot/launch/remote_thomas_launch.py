from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    thomas_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('thomas'),
                'launch',
                'brickpi3_motors_launch.py'
            ])
        ]),
    )

    camera_node = Node(
        package='image_tools',
        executable='cam2image',
        parameters=[{"frequency": 10.0}],
        remappings=[
            ('/image', '/thomas/image')]
    )
    compress_node = Node(
        package='image_transport',
        executable='republish',
#        arguments=['raw', 'compressed'],
        parameters=[{"in_transport": "raw", "out_transport": "compressed"}],
        remappings=[
            ('in', '/thomas/image'),
            ('out/compressed', '/thomas/compressed')]
    )
    nodes = [
        thomas_launch,
        camera_node,
        compress_node
    ]

    return LaunchDescription(nodes)

