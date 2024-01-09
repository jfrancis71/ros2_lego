from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ev3_pi',
            executable='differential_drive_node',
        ),
        Node(
            package='ev3_pi',
            executable='compass_node',
        ),
        Node(
            package='ev3_pi',
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
