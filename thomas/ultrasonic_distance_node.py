import brickpi3
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range


class UltrasonicDistancePublisher(Node):
    def __init__(self):
        super().__init__("ultrasonic_distance_node")
        self.publisher = self.create_publisher(Range, "ultrasonic_distance", 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ultrasonic_distance = bp.get_sensor(bp.PORT_2)
        msg = Range()
        msg.header.frame_id = "ultrasonic_distance_sensor"
        msg.radiation_type = 0  # ULTRASONIC TODO use constant to improve code style?
        msg.field_of_view = 0.1  # need to check
        msg.min_range = 0.0
        msg.max_range = 1.0
        msg.range = ultrasonic_distance/100.0  # raw sensor is in cm's
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.range)


bp = brickpi3.BrickPi3()
bp.set_sensor_type(bp.PORT_2, bp.SENSOR_TYPE.EV3_ULTRASONIC_CM)
rclpy.init()
ultrasonic_distance_publisher = UltrasonicDistancePublisher()
rclpy.spin(ultrasonic_distance_publisher)
ultrasonic_distance_publisher.destroy_node()
rclpy.shutdown()
