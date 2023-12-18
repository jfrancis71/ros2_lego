import brickpi3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt16


class CompassPublisher(Node):
    def __init__(self):
        super().__init__("compass_node")
        self.publisher = self.create_publisher(UInt16, "compass_direction", 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        bp.transact_i2c(bp.PORT_1, 0b00000010, [0x42], 2) # Ref 1
        time.sleep(.01)
        value = bp.get_sensor(bp.PORT_1)
        compass_direction = value[0]*2 + value[1]
        msg = UInt16()
        msg.data = compass_direction
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)


bp = brickpi3.BrickPi3()
bp.set_sensor_type(bp.PORT_1, bp.SENSOR_TYPE.I2C, [0,20]) # Ref 1
rclpy.init()
compass_publisher = CompassPublisher()
rclpy.spin(compass_publisher)
compass_publisher.destroy_node()
rclpy.shutdown()


# Ref 1: https://forum.dexterindustries.com/t/hitechnic-compass-sensor-with-brickpi3/7906
