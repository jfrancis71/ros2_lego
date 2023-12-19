from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_transport',
            executable='republish',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', '/thomas/compressed'),
                ('out', '/image')]
        ),
        Node(
            package='thomas',
            executable='object_detector_node'
        ),
        Node(
            package='thomas',
            executable='person_follower_node'
        )
    ])
