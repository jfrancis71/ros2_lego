from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_transport',
            executable='republish',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', '/terry/compressed'),
                ('out', '/image')]
        ),
    ])
