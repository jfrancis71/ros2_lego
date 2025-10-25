import time
import copy
import itertools
import os
import yaml
import skimage
import numpy as np
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
        self.declare_parameter('map', 'my_house.yaml')
        self.map_file = self.get_parameter('map').get_parameter_value().string_value
        with open(self.map_file, 'r') as map_file:
            map_properties = yaml.safe_load(map_file)
            self.image_filename = map_properties['image']
            self.origin = map_properties['origin']
            self.resolution = map_properties["resolution"]
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.pred_publisher = \
            self.create_publisher(LaserScan, "/pred_laser", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.map = skimage.io.imread(os.path.join(os.path.split(self.map_file)[0], self.image_filename))
        self.centers = [[(r,c) for c in range(0, self.map.shape[1], 10)] for r in range(0, self.map.shape[0], 10)]
        self.centers = list(itertools.chain(*self.centers))
        self.trans = np.array([skimage.transform.warp_polar(self.map==0, center=c, radius=100) for c in self.centers])
        self.polar_coords = np.array([np.argmax(self.trans[c], axis=1)*self.resolution for c in range(len(self.centers))])

    def send_map_base_link_transform(self, loc, angle, tim):
        try:
            base_link_to_odom_transform = self.tf_buffer.lookup_transform(
                "base_link",
                "odom",
                rclpy.time.Time())
        except TransformException as ex:
            print("No Transform")
            return
        try:
            base_laser_to_base_link_transform = self.tf_buffer.lookup_transform(
                "base_laser",
                "base_link",
                rclpy.time.Time())
        except TransformException as ex:
            print("No Transform")
            return
        zero_to_odom_transform = TransformStamped()
        zero_to_odom_transform.header.stamp = self.get_clock().now().to_msg()
        zero_to_odom_transform.header.frame_id = 'zero'
        zero_to_odom_transform.child_frame_id = 'odom'
        zero_to_odom_transform.transform = base_link_to_odom_transform.transform
        self.tf_broadcaster.sendTransform(zero_to_odom_transform)

        map_base_laser_to_zero_transform = TransformStamped()
        map_base_laser_to_zero_transform.header.stamp = \
            self.get_clock().now().to_msg()
        map_base_laser_to_zero_transform.header.frame_id = 'map_base_laser'
        map_base_laser_to_zero_transform.child_frame_id = 'zero'
        map_base_laser_to_zero_transform.transform = \
            base_laser_to_base_link_transform.transform
        self.tf_broadcaster.sendTransform(map_base_laser_to_zero_transform)

        map_to_map_base_laser_transform = TransformStamped()
        map_to_map_base_laser_transform.header.stamp = \
            self.get_clock().now().to_msg()
        map_to_map_base_laser_transform.header.frame_id = 'map'
        map_to_map_base_laser_transform.child_frame_id = 'map_base_laser'
        map_to_map_base_laser_transform.transform.translation.x = \
            self.centers[loc][1]*self.resolution + self.origin[0]
        map_to_map_base_laser_transform.transform.translation.y = \
            (self.map.shape[0]-self.centers[loc][0])*self.resolution + self.origin[1]
        map_to_map_base_laser_transform.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, -angle)
        map_to_map_base_laser_transform.transform.rotation.x = q[0]
        map_to_map_base_laser_transform.transform.rotation.y = q[1]
        map_to_map_base_laser_transform.transform.rotation.z = q[2]
        map_to_map_base_laser_transform.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(map_to_map_base_laser_transform)


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
        loc = idx[0]
        angle = 2 * np.pi * idx[1]/360.0
        self.send_map_base_link_transform(loc, angle, None)
        self.publish_lidar_prediction(lidar_msg.header, predictions[loc][idx[1]])


rclpy.init()
localizer = Localizer()
rclpy.spin(localizer)
nav.destroy_node()
rclpy.shutdown()
