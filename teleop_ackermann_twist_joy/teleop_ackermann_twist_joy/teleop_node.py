import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped


class AckermannJoystickNode(Node):
    def __init__(self):
        super().__init__("teleop_ackermann_twist_joy_node")
#        super().__init__("teleop_node")
        self.joy_subscription = self.create_subscription(
            Joy,
            "/joy",
            self.joy_callback,
            1)
        self.twist_publisher = self.create_publisher(TwistStamped, '/cmd_vel', 1)
        self.declare_parameter('axis_linear.x', 1)
        self.axis_linear = self.get_parameter('axis_linear.x').get_parameter_value().integer_value
        self.declare_parameter('scale_linear.x', 1.0)
        self.scale_linear = self.get_parameter('scale_linear.x').get_parameter_value().double_value
        self.declare_parameter('scale_linear_turbo.x', 1.0)
        self.scale_linear = self.get_parameter('scale_linear_turbo.x').get_parameter_value().double_value
        self.declare_parameter('axis_angular.yaw', 1)
        self.axis_angular = self.get_parameter('axis_angular.yaw').get_parameter_value().integer_value
        self.declare_parameter('min_turn', 1.0)
        self.min_turn = self.get_parameter('min_turn').get_parameter_value().double_value
        self.declare_parameter('enable_button', 8)
        self.enable_button = self.get_parameter('enable_button').get_parameter_value().integer_value
        self.declare_parameter('enable_turbo_button', 10)
        self.enable_turbo_button = self.get_parameter('enable_turbo_button').get_parameter_value().integer_value
        self.moving = False

    def joy_callback(self, callback):
        linear_x = 0.0
        moving = False
        if callback.buttons[self.enable_button] == 1:
            linear_x = callback.axes[self.axis_linear] * self.scale_linear
            moving = True
        elif callback.buttons[self.enable_turbo_button] == 1:
            linear_x = callback.axes[self.axis_linear] * self.scale_linear_turbo
            moving = True
        if moving == True or self.moving == True:
            twist_stamped_msg = TwistStamped()
            twist_stamped_msg.header.stamp = callback.header.stamp
            twist_stamped_msg.header.frame_id = "teleop_ackermann_twist_joy"
            twist_stamped_msg.twist.linear.x = linear_x
            twist_stamped_msg.twist.angular.z = callback.axes[self.axis_angular] * linear_x / self.min_turn
            self.twist_publisher.publish(twist_stamped_msg)
            if moving == False:
                self.moving = False
            else:
                self.moving = True


rclpy.init()
ackermann_joystick_node = AckermannJoystickNode()
rclpy.spin(ackermann_joystick_node)
ackermann_joystick_node.destroy_node()
rclpy.shutdown()
