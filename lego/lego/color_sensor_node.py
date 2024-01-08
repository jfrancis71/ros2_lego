"""ColorSensorNode supports 3 light detection modes:
   COLOR: Returns one of none, Black, Blue, Green, Yellow, Red, White, Brown.
   REFLECTED: Returns UInt8 between 0-100 indicating percentage of reflected light received.
   AMBIENT: Returns UInt8 between 0-100 of ambient light received
       (ie sensor does not activate illumination).
"""

import brickpi3
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8
from std_msgs.msg import String


class ColorSensorNode(Node):
    """ColorSensorNode publishes appropriate message"""
    def __init__(self):
        super().__init__("color_sensor_node")
        self.bp = brickpi3.BrickPi3()
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
        self.declare_parameter('detection_mode', 'color_mode')
        self.detection_mode = \
            self.get_parameter('detection_mode').get_parameter_value().string_value
        if self.detection_mode == "COLOR":
            self.bp.set_sensor_type(self.lego_port, self.bp.SENSOR_TYPE.EV3_COLOR_COLOR)  # pylint: disable=E1101
            self.publisher = self.create_publisher(String, "color", 10)
        else:
            if self.detection_mode in ( "REFLECTED", "AMBIENT"):
                self.publisher = self.create_publisher(UInt8, "light_intensity", 10)
                if self.detection_mode == "REFLECTED":
                    self.bp.set_sensor_type(self.lego_port, self.bp.SENSOR_TYPE.EV3_COLOR_REFLECTED)  # pylint: disable=E1101
                else:
                    self.bp.set_sensor_type(self.lego_port, self.bp.SENSOR_TYPE.EV3_COLOR_AMBIENT)  # pylint: disable=E1101
            else:
                error_msg = f'Unknown mode: {self.detection_mode}'
                self.get_logger().fatal(error_msg)
                raise ValueError(error_msg)
        self.colormap = ["none", "Black", "Blue", "Green", "Yellow", "Red", "White", "Brown"]
        self.declare_parameter('frequency', 2.0)
        timer_period = 1.0/self.get_parameter('frequency').get_parameter_value().double_value
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        """reads color sensor and publishes appropriate message"""
        try:
            value = self.bp.get_sensor(self.lego_port)
        except brickpi3.SensorError as e:
            error_msg = f'Invalid color sensor data on {self.lego_port_name}'
            self.get_logger().error(error_msg)
            raise brickpi3.SensorError(error_msg) from e
        if self.detection_mode == "COLOR":
            msg = String()
            if value > len(self.colormap):
                error_msg = f'Invalid color returned from {self.lego_port_name}'
                self.get_logger().error(error_msg)
                raise brickpi3.SensorError(error_msg)
            msg.data = self.colormap[value]
        else:
            msg = UInt8()
            msg.data = value
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')


rclpy.init()
color_sensor_node = ColorSensorNode()
rclpy.spin(color_sensor_node)
color_sensor_node.destroy_node()
rclpy.shutdown()
