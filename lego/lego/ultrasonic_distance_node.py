"""Ultrasonic Distance Sensor Node"""
import brickpi3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range


class UltrasonicDistanceNode(Node):
    """Publishes Range message on topic ultrasonic_distance"""
    def __init__(self):
        super().__init__("ultrasonic_distance_node")
        self.bp = brickpi3.BrickPi3()
        self.publisher = self.create_publisher(Range, "ultrasonic_distance", 10)
        self.declare_parameter('lego_port', 'PORT_1')
        port_dict = { "PORT_1": self.bp.PORT_1,
              "PORT_2": self.bp.PORT_2,
              "PORT_3": self.bp.PORT_3,
              "PORT_4": self.bp.PORT_4 }
        lego_port_name = self.get_parameter('lego_port').get_parameter_value().string_value
        self.lego_port = port_dict[lego_port_name]
        self.bp.set_sensor_type(self.lego_port, self.bp.SENSOR_TYPE.EV3_ULTRASONIC_CM)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        """Reads ultrasonic distance and publishes Range message on topic ultrasonic_distance"""
        ultrasonic_distance = self.bp.get_sensor(self.lego_port)
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ultrasonic_distance_sensor"
        msg.radiation_type = 0  # ULTRASONIC TODO use constant to improve code style?
        msg.field_of_view = 0.05  # very approximate
        msg.min_range = 0.0
        msg.max_range = 1.0
        msg.range = ultrasonic_distance/100.0  # raw sensor is in cm's
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.range}')


rclpy.init()
ultrasonic_distance_node = UltrasonicDistanceNode()
rclpy.spin(ultrasonic_distance_node)
ultrasonic_distance_node.destroy_node()
rclpy.shutdown()
