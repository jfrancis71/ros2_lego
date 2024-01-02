from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lego',
            executable='charlie_node',
        ),
        Node(
            package='lego',
            executable='compass_node',
        ),
        Node(
            package='lego',
            executable='ultrasonic_distance_node',
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
