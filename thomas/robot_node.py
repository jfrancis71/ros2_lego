import time
import brickpi3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class ThomasNode(Node):
    def __init__(self):
        super().__init__('thomas_the_robot')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.listener_callback, 0)
        self.bp = brickpi3.BrickPi3()
        self.speed = 50
        timer_period = 0.2
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.last_command_time = time.time()
        self.off()
        print("Robot ready.")

    def listener_callback(self, msg):
        print("Callback message: ", msg)
        self.last_command_time = time.time()
        motora = msg.linear.x*25
        motorb = msg.linear.x*25
        motora -= msg.angular.z * 10
        motorb += msg.angular.z * 10
        self.bp.set_motor_power(self.bp.PORT_A, motora)
        self.bp.set_motor_power(self.bp.PORT_D, motorb)

    def timer_callback(self):
        timeout = .5
        if (time.time() - self.last_command_time) > timeout:
            self.off()

    def off(self):
        self.bp.set_motor_power(self.bp.PORT_A, 0)
        self.bp.set_motor_power(self.bp.PORT_D, 0)

rclpy.init()
motor_subscriber = ThomasNode()
rclpy.spin(motor_subscriber)
motor_subscriber.off()
motor_subscriber.destroy_node()
rclpy.shutdown()
