"""Gyro Sensor Node"""
import math
import brickpi3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class GyroNode(Node):
    """Publishes Float64 message on /gyro_angular_velocity"""
    def __init__(self):
        super().__init__("gyro_node")
        self.bp = brickpi3.BrickPi3()
        self.publisher = self.create_publisher(Float64, "gyro_angular_velocity", 10)
        self.declare_parameter('lego_port', 'PORT_1')
        port_dict = { "PORT_1": self.bp.PORT_1,
              "PORT_2": self.bp.PORT_2,
              "PORT_3": self.bp.PORT_3,
              "PORT_4": self.bp.PORT_4 }
        self.lego_port_name = self.get_parameter('lego_port').get_parameter_value().string_value
        try:
            self.lego_port = port_dict[self.lego_port_name]
        except KeyError as e:
            error_msg = f'Unknown lego input port: {e}'
            self.get_logger().fatal(error_msg)
            raise IOError(error_msg) from e
        self.lego_port = port_dict[self.lego_port_name]
        # we disable pylint warning as BrickPi does some strange attribute manipulation
        self.bp.set_sensor_type(self.lego_port, self.bp.SENSOR_TYPE.EV3_GYRO_ABS_DPS)  # pylint: disable=E1101
        self.declare_parameter('frequency', 2.0)
        timer_period = 1.0/self.get_parameter('frequency').get_parameter_value().double_value
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        """Reads gyro angular velocity and publishes on /gyro_angular_velocity"""
        try:
            _, gyro_angular_velocity = self.bp.get_sensor(self.lego_port)
        except brickpi3.SensorError as e:
            error_msg = f'Invalid gyro sensor data on {self.lego_port_name}'
            self.get_logger().error(error_msg)
            raise brickpi3.SensorError(error_msg) from e
        msg = Float64()
        msg.data = gyro_angular_velocity*(2*math.pi/360)
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')


rclpy.init()
gyro_node = GyroNode()
rclpy.spin(gyro_node)
gyro_node.destroy_node()
rclpy.shutdown()
