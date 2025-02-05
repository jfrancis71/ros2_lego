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
            package='coco_detector',
            executable='coco_detector_node'
        ),
        Node(
            package='charlie',
            executable='person_follower_node'
        )
    ])
