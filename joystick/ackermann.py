import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped


class AckermannJoystickNode(Node):
    def __init__(self):
        super().__init__("teleop_ackermann_twist_joy_node")
        self.lidar_subscription = self.create_subscription(
            Joy,
            "/joy",
            self.joy_callback,
            1)
        self.twist_publisher = self.create_publisher(TwistStamped, '/cmd_vel', 1)

    def joy_callback(self, callback):
#        L = 0.08
        R = 0.25
        print("Callback=", callback)
        if callback.buttons[4] == 1:
            twist_stamped_msg = TwistStamped()
            twist_stamped_msg.header.stamp = callback.header.stamp
            twist_stamped_msg.header.frame_id = "teleop_ackermann_twist_joy"
            twist_stamped_msg.twist.linear.x = callback.axes[1] * 0.1
#            twist_stamped_msg.twist.angular.z = math.sin(callback.axes[2] * 0.5) * twist_stamped_msg.twist.linear.x / L
            twist_stamped_msg.twist.angular.z = callback.axes[2] * twist_stamped_msg.twist.linear.x / R
            self.twist_publisher.publish(twist_stamped_msg)


rclpy.init()
ackermann_joystick_node = AckermannJoystickNode()
rclpy.spin(ackermann_joystick_node)
ackermann_joystick_node.destroy_node()
rclpy.shutdown()
