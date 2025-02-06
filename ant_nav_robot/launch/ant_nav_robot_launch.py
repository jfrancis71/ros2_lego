from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_description_content = Command([
        "cat ",
        PathJoinSubstitution(
            [FindPackageShare("terry"), "config", "robot_hardware_description.urdf"])
        ])
    robot_controllers = PathJoinSubstitution(
            [FindPackageShare("terry"), "config", "robot_description.yaml"])
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[{"robot_description": robot_description_content}, robot_controllers],
        remappings=[("/omni_wheel_controller/cmd_vel_unstamped", "/cmd_vel")],
        output="both",
    )
    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["omni_wheel_controller", "--controller-manager", "/controller_manager"],
    )
    camera_node = Node(
        package='image_tools',
        executable='cam2image',
        remappings=[
            ('/image', '/terry/image')]
    )
    compress_node = Node(
        package='image_transport',
        executable='republish',
        arguments=['raw', 'compressed'],
        remappings=[
            ('in', '/terry/image'),
            ('out/compressed', '/terry/compressed')]
    )
    nodes = [
        control_node,
        robot_controller_spawner,
        camera_node,
        compress_node
    ]

    return LaunchDescription(nodes)

