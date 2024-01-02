"""CompassNode publishes message UInt16 on topic compass_direction"""
import time
import brickpi3
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt16


class CompassNode(Node):
    """CompassNode publishes message UInt16 on topic compass_direction"""
    def __init__(self):
        super().__init__("compass_node")
        self.publisher = self.create_publisher(UInt16, "compass_direction", 10)
        self.declare_parameter('lego_port', 'PORT_1')
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.bp = brickpi3.BrickPi3()
        port_dict = { "PORT_1": self.bp.PORT_1,
                      "PORT_2": self.bp.PORT_2,
                      "PORT_3": self.bp.PORT_3,
                      "PORT_4": self.bp.PORT_4 }
        lego_port_name = self.get_parameter('lego_port').get_parameter_value().string_value
        self.lego_port = port_dict[lego_port_name]
        self.bp.set_sensor_type(self.lego_port, self.bp.SENSOR_TYPE.I2C, [0,20]) # Ref 1

    def timer_callback(self):
        """reads compass direction and publishes on topic compass_direction"""
        self.bp.transact_i2c(self.lego_port, 0b00000010, [0x42], 2) # Ref 1
        time.sleep(.01)
        value = self.bp.get_sensor(self.lego_port)
        compass_direction = value[0]*2 + value[1]
        msg = UInt16()
        msg.data = compass_direction
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')


rclpy.init()
compass_node = CompassNode()
rclpy.spin(compass_node)
compass_node.destroy_node()
rclpy.shutdown()


# Ref 1: https://forum.dexterindustries.com/t/hitechnic-compass-sensor-with-brickpi3/7906
