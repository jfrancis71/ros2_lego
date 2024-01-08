"""CharlieNode listens to Twist messages and drives Charlie the robot"""
import math
import brickpi3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState


class CharlieNode(Node):
    """CharlieNode listens to Twist messages and drives Charlie the robot"""
    def __init__(self):
        super().__init__('charlie_the_robot')
        self.bp = brickpi3.BrickPi3()
        self.off()
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.listener_callback, 0)
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.wheel_radius = 0.02
        self.turning_circle_radius = 0.132/2.0
        self.declare_parameter('joint_states_publish_frequency', 5.0)
        joint_states_timer_period = \
            1.0/self.get_parameter('joint_states_publish_frequency'). \
            get_parameter_value().double_value
        self.joint_states_times = \
            self.create_timer(joint_states_timer_period, self.joint_states_callback)
        self.declare_parameter('timeout', 1.0)
        timeout = self.get_parameter('timeout').get_parameter_value().double_value
        self.timeout_timer = self.create_timer(timeout, self.timeout_callback)
        self.get_logger().info('Robot ready.')

    def listener_callback(self, msg):
        """sends lego drive commands to robot in response to Twist messages"""
        translation_speed_dps = (360.0/(2*math.pi))*msg.linear.x/self.wheel_radius
        rotation_speed_dps = (360.0/(2*math.pi))* \
            msg.angular.z*self.turning_circle_radius/self.wheel_radius
        motora_dps = translation_speed_dps - rotation_speed_dps
        motord_dps = translation_speed_dps + rotation_speed_dps
        self.bp.set_motor_dps(self.bp.PORT_A, motora_dps)
        self.bp.set_motor_dps(self.bp.PORT_D, motord_dps)
        self.timeout_timer.reset()

    def joint_states_callback(self):
        """publish the positions of the wheels"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "charlie"
        msg.name = ["left_wheel", "right_wheel"]
        left_wheel_pos = float(self.bp.get_motor_encoder(self.bp.PORT_A))
        right_wheel_pos = float(self.bp.get_motor_encoder(self.bp.PORT_D))
        msg.position = [left_wheel_pos, right_wheel_pos]
        self.publisher.publish(msg)

    def timeout_callback(self):
        """stop the robot if no Twist message received within timeout"""
        self.off()

    def off(self):
        """stop the robot"""
        self.bp.set_motor_power(self.bp.PORT_A, 0)
        self.bp.set_motor_power(self.bp.PORT_D, 0)


rclpy.init()
charlie_node = CharlieNode()
rclpy.spin(charlie_node)
charlie_node.off()
charlie_node.destroy_node()
rclpy.shutdown()
