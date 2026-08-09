import copy
import os
import yaml
import numpy as np
#import scipy
from scipy import ndimage as ndi
from scipy.stats import uniform, norm, vonmises
from scipy.spatial.transform import Rotation as R
import scipy.stats as stats
import skimage
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped, PoseWithCovarianceStamped
from tf2_ros.transform_listener import TransformListener
# Current StaticTransformBroadcaster is broken, we need to use from rolling.
# clone git clone https://github.com/ros2/geometry2.git
# Prepend ./src/geometry2/tf2_ros_py/tf2_ros to PYTHONPATH and export
from static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros.buffer import Buffer
import time
import message_filters
from nav_msgs.msg import OccupancyGrid
#std_srvs/srv/Empty
from std_srvs.srv import Empty
import atexit


def angle_diff(angle_1, angle_2):
    """Returns closest difference between two angles.

       Note you cannot just subtract the angles, eg 3.1 - (-3.1) = 6.2. This not the closest angle change.

    """
    abs_diff_angle = np.abs(angle_1 - angle_2)
    return np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))


class RobotDataCollectorNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.min_dist = 0.03  # minimum distance for lidar update
        self.min_angle = .08  # minimum angle change for lidar update
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)

        self.tf_buffer = Buffer()
        qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True, qos=qos)
        self.current_lidar_msg = None
        self.previous_odom_pose = None
        self.init_wait = 0
        self.declare_parameter('filename', 'odom.npz')
        self.filename = self.get_parameter('filename').get_parameter_value().string_value
        self.scans = []
        self.odom_delta = []
        self.num_angles = 360

    def robot_frame_odom(self, previous_odom_pose, current_odom_pose):
        diff_x = current_odom_pose[0] - previous_odom_pose[0]
        diff_y = current_odom_pose[1] - previous_odom_pose[1]
        forward = diff_x * np.cos(previous_odom_pose[2]) + diff_y * np.sin(previous_odom_pose[2])
        slip = diff_x * np.cos(previous_odom_pose[2] + np.pi/2) + diff_y * np.sin(previous_odom_pose[2] + np.pi/2)
        diff_angle = angle_diff(current_odom_pose[2], previous_odom_pose[2])
        d_rot = current_odom_pose[2] - previous_odom_pose[2]
        print("forward=", forward, " slip=", slip, d_rot)
        return forward, slip, d_rot

    def ros2_to_pose(self, odom_transform):
        trans_tf = odom_transform.transform.translation
        rot_tf = odom_transform.transform.rotation
        rot = [rot_tf.x, rot_tf.y, rot_tf.z, rot_tf.w]
        _, _, theta = euler_from_quaternion(rot)
        return (trans_tf.x, trans_tf.y, theta)

    def robot_moved(self, current_odom_pose):
        diff_x = current_odom_pose[0] - self.previous_odom_pose[0]
        diff_y = current_odom_pose[1] - self.previous_odom_pose[1]
        d_trans = np.sqrt(diff_y**2 + diff_x**2)
        diff_angle = angle_diff(current_odom_pose[2], self.previous_odom_pose[2])
        return np.abs(diff_angle) > self.min_angle or d_trans > self.min_dist

    def lidar_callback(self, lidar_msg):
        if self.init_wait < 10:
            self.init_wait += 1
            return
        if self.current_lidar_msg is None:
            self.current_lidar_msg = lidar_msg
            return
        lidar_msg_time = Time.from_msg(self.current_lidar_msg.header.stamp)
        try:
            tf_base_laser_to_odom = self.tf_buffer.lookup_transform(
                "base_laser",
                "odom",
                lidar_msg_time)  # https://github.com/ros2/ros2_documentation/issues/4385
            tf_odom_to_base_laser = self.tf_buffer.lookup_transform(
                "odom",
                "base_laser",
                lidar_msg_time)
        except TransformException as ex:  # This is common and normal.
            return
        self.process_lidar(self.current_lidar_msg, tf_base_laser_to_odom, tf_odom_to_base_laser)
        self.current_lidar_msg = None

    def process_lidar(self, lidar_msg, tf_base_laser_to_odom, tf_odom_to_base_laser):
        scan = np.array(lidar_msg.ranges).astype(np.float32)
        current_odom_pose = self.ros2_to_pose(tf_odom_to_base_laser)
        if self.init_wait == 10:
            self.scans.append(np.array(skimage.transform.resize(scan, (self.num_angles,))))
            self.init_wait += 1
            return
        if self.previous_odom_pose is None:
            self.previous_odom_pose = current_odom_pose
        if self.robot_moved(current_odom_pose):
            robot_frame_odom = self.robot_frame_odom(self.previous_odom_pose, current_odom_pose)
            self.scans.append(np.array(skimage.transform.resize(scan, (self.num_angles,))))
            self.odom_delta.append(robot_frame_odom)
            self.previous_odom_pose = current_odom_pose

    def save(self):
        np.savez(self.filename, odom=np.array(self.odom_delta), scans=np.array(self.scans))


rclpy.init()
robot_data_collector_node = RobotDataCollectorNode()
atexit.register(robot_data_collector_node.save)
rclpy.spin(robot_data_collector_node)
robot_data_collector_node.destroy_node()
rclpy.shutdown()
