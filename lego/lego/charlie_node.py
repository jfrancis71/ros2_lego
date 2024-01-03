"""CharlieNode listens to Twist messages and drives Charlie the robot"""
import time
import math
import brickpi3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CharlieNode(Node):
    """CharlieNode listens to Twist messages and drives Charlie the robot"""
    def __init__(self):
        super().__init__('charlie_the_robot')
        self.bp = brickpi3.BrickPi3()
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.listener_callback, 0)
        self.off()
        self.wheel_radius = 0.02
        self.turning_circle_radius = 0.132/2.0
        timer_period = 0.2
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.last_command_time = time.time()
        self.get_logger().info('Robot ready.')

    def listener_callback(self, msg):
        """sends lego drive commands to robot in response to Twist messages"""
        self.last_command_time = time.time()
        translation_speed_dps = (360.0/(2*math.pi))*msg.linear.x/self.wheel_radius
        rotation_speed_dps = (360.0/(2*math.pi))* \
            msg.angular.z*self.turning_circle_radius/self.wheel_radius
        motora_dps = translation_speed_dps - rotation_speed_dps
        motord_dps = translation_speed_dps + rotation_speed_dps
        self.bp.set_motor_dps(self.bp.PORT_A, motora_dps)
        self.bp.set_motor_dps(self.bp.PORT_D, motord_dps)

    def timer_callback(self):
        """stop the robot if no Twist message received within timeout"""
        timeout = .5
        if (time.time() - self.last_command_time) > timeout:
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
