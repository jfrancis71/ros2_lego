from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='thomas',
            executable='compass_node',
        ),
        Node(
            package='thomas',
            executable='robot_node',
        ),
        Node(
            package='image_tools',
            executable='cam2image'
        ),
        Node(
            package='image_transport',
            executable='republish',
            arguments=['raw', 'compressed'],
            remappings=[
                ('in', '/image'),
                ('out/compressed', '/thomas/compressed')]
        ),
    ])

