from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_transport',
            executable='republish',
            arguments=['compressed', 'raw'],
            remappings=[
                ('in/compressed', '/charlie/compressed'),
                ('out', '/image')]
        ),
        Node(
            package='lego',
            executable='object_detector_node'
        ),
        Node(
            package='lego',
            executable='person_follower_node'
        )
    ])
