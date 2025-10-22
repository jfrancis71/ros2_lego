import rclpy
import skimage
import numpy as np
import itertools
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException


class Localizer(Node):
    def __init__(self):
        super().__init__("nav")
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.pred_publisher = \
            self.create_publisher(LaserScan, "/pred_laser", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.house = skimage.io.imread("~/ros2_ws/my_house.pgm")
        print("Dims=", self.house.shape)
        self.centers = [[(r,c) for c in range(0, 317, 10)] for r in range(0, 118, 10)]
        self.centers = list(itertools.chain(*self.centers))
        self.trans = np.array([skimage.transform.warp_polar(self.house==0, center=c, radius=100) for c in self.centers])
        self.polar_coords = np.array([np.argmax(self.trans[c], axis=1)*0.05 for c in range(len(self.centers))])
        self.tf_broadcaster = TransformBroadcaster(self)

    def send_map_base_link_transform(self, loc, angle, tim):
    # This needs checking carefully, works for a lidar mounted clockwise 90 degrees
    # from base link
    # Also this does not work with odom frame. Ideally should work with map->odom->base_link->base_laser
        try:
            lookup = self.tf_buffer.lookup_transform(
                "base_link",
                "base_laser",
                rclpy.time.Time())
        except TransformException as ex:
            print("No Transform")
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.centers[loc][1]*.05 -13.640
        t.transform.translation.y = (118-self.centers[loc][0])*0.05 -3.959
        t.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, -angle + np.pi/2)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    def publish_lidar_prediction(self, header, ranges):
        lidar_msg1 = LaserScan()
        lidar_msg1.ranges = ranges
        lidar_msg1.angle_min = 0.0
        lidar_msg1.angle_max = 6.28318548
        lidar_msg1.angle_increment = 2 * np.pi / 360.
        lidar_msg1.time_increment = 0.00019850002718158066
        lidar_msg1.scan_time = 0.10004401206970215
        lidar_msg1.range_min = 0.019999999552965164
        lidar_msg1.range_max = 25.0
        lidar_msg1.header = header
        self.pred_publisher.publish(lidar_msg1)

    def lidar_callback(self, lidar_msg):
        scan = np.array(lidar_msg.ranges)
        new_scan = skimage.transform.resize(scan.astype(np.float32), (360,))
        predictions = np.array([[np.roll(np.flip(self.polar_coords[c]), s) for s in range(359)] for c in range(len(self.centers))])

        angles = np.nanmean((predictions - new_scan)**2, axis=0)[np.newaxis, np.newaxis]
        prediction_error = np.nanmean((predictions - new_scan)**2, axis=2)
        idx = np.unravel_index(np.argmin(prediction_error), prediction_error.shape)
        print("idx=", idx)
        loc = idx[0]
        angle = 2 * np.pi * idx[1]/360.0
        self.send_map_base_link_transform(loc, angle, None)
        self.publish_lidar_prediction(lidar_msg.header, predictions[loc][idx[1]])


rclpy.init()
localizer = Localizer()
rclpy.spin(localizer)
nav.destroy_node()
rclpy.shutdown()
