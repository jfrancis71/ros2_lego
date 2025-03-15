from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_description_content = Command([
        "cat ",
        PathJoinSubstitution(
            [FindPackageShare("thomas"), "config", "robot_hardware_description.urdf"])
        ])
    robot_controllers = PathJoinSubstitution(
            [FindPackageShare("thomas"), "config", "robot_description.yaml"])
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[{"robot_description": robot_description_content}, robot_controllers],
        remappings=[("/differential_drive_controller/cmd_vel_unstamped", "/cmd_vel")]
,
        output="both",
    )
    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["differential_drive_controller", "--controller-manager", "/controller_manager"],
    )
    camera_node = Node(
        package='image_tools',
        executable='cam2image',
        remappings=[
            ('/image', '/thomas/image')]
    )
    compress_node = Node(
        package='image_transport',
        executable='republish',
        arguments=['raw', 'compressed'],
        remappings=[
            ('in', '/thomas/image'),
            ('out/compressed', '/thomas/compressed')]
    )
    nodes = [
        control_node,
        robot_controller_spawner,
        camera_node,
        compress_node
    ]

    return LaunchDescription(nodes)

