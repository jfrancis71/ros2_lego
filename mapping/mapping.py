import math
import numpy as np
import rclpy
import scipy.stats as stats
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point, TransformStamped, PoseWithCovarianceStamped
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf_transformations import euler_from_quaternion


def angle_diff(angle_1, angle_2):
    """Returns closest difference between two angles.

       Note you cannot just subtract the angles, eg 3.1 - (-3.1) = 6.2. This not the closest angle change.

    """
    abs_diff_angle = np.abs(angle_1 - angle_2)
    return np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))


class MappingNode(Node):
    def __init__(self):
        super().__init__("mapping_node")
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.map_publisher = \
            self.create_publisher(OccupancyGrid, "my_map", 1)
        self.prior_occupancy_prob = 0.2
        self.tf_buffer = Buffer()
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True, qos=qos)
        self.initial_pose_received = False
        self.current_lidar_msg = None
        self.initialpose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.initialpose_callback,
            1)
        self.neg= np.zeros([118, 317]) + 2
        self.pos= np.zeros([118, 317]) + 0.5
        self.previous_odom_pose = None
        self.min_dist = 0.03  # minimum distance for lidar update
        self.min_angle = .08  # minimum angle change for lidar update

    def initialpose_callback(self, initialpose_msg):
        try:
            tf_base_link_to_base_laser = self.tf_buffer.lookup_transform(
                "base_link",
                "base_laser",
                Time())
        except TransformException as ex:
            warn_msg = "No Transform for base_link to base_laser. Initial pose not set."
            self.get_logger().warn(warn_msg)
            return
        print("POSE SET")
        self.initial_pose_received = True

    def robot_moved(self, current_odom_pose):
        diff_x = current_odom_pose[0] - self.previous_odom_pose[0]
        diff_y = current_odom_pose[1] - self.previous_odom_pose[1]
        d_trans = np.sqrt(diff_y**2 + diff_x**2)
        diff_angle = angle_diff(current_odom_pose[2], self.previous_odom_pose[2])
        return np.abs(diff_angle) > self.min_angle or d_trans > self.min_dist

    def lidar_callback(self, lidar_msg):
        if not self.initial_pose_received:
            return
        if self.current_lidar_msg is None:
            self.current_lidar_msg = lidar_msg
            return
        lidar_msg_time = Time.from_msg(self.current_lidar_msg.header.stamp)
        try:
            tf_base_link_to_map = self.tf_buffer.lookup_transform(
                "map",
                "base_laser",
                lidar_msg_time)  # https://github.com/ros2/ros2_documentation/issues/4385
        except TransformException as ex:  # This is common and normal.
            print("NO MAP")
            return
        try:
            tf_base_link_to_base_laser = self.tf_buffer.lookup_transform(
                "base_link",
                "base_laser",
                Time())
        except TransformException as ex:
            warn_msg = "No Transform for base_link to base_laser. Initial pose not set."
            self.get_logger().warn(warn_msg)
            return
        print("OIJ")
        self.process_lidar(lidar_msg, tf_base_link_to_map)
        self.current_lidar_msg = None

    def ros2_to_pose(self, odom_transform):
        trans_tf = odom_transform.transform.translation
        rot_tf = odom_transform.transform.rotation
        rot = [rot_tf.x, rot_tf.y, rot_tf.z, rot_tf.w]
        _, _, theta = euler_from_quaternion(rot)
        return (trans_tf.x, trans_tf.y, theta)

    def process_lidar(self, lidar_msg, tf_base_link_to_map):
        current_odom_pose = self.ros2_to_pose(tf_base_link_to_map)
        if self.previous_odom_pose is None:
            self.previous_odom_pose = current_odom_pose
        if not self.robot_moved(current_odom_pose):
            return
        self.previous_odom_pose = current_odom_pose
        my_map_msg = OccupancyGrid()
        my_map_msg.header = lidar_msg.header
        my_map_msg.header.frame_id = "map"
        my_map_msg.info.resolution = .05
        my_map_msg.info.width = 317
        my_map_msg.info.height = 118
        my_map_msg.info.origin.position.x = -13.640
        my_map_msg.info.origin.position.y = -3.959
        print("X=", tf_base_link_to_map.transform.translation.x)
        rot_tf = tf_base_link_to_map.transform.rotation
        rot = [rot_tf.x, rot_tf.y, rot_tf.z, rot_tf.w]
        _, _, theta = euler_from_quaternion(rot)
        print("THETA=", theta)
        for x in range(317):
            for y in range(118):
                row = 118 - int((tf_base_link_to_map.transform.translation.y-(-3.959))*(1/.05))
                row = int((tf_base_link_to_map.transform.translation.y-(-3.959))*(1/.05))
                col = int((tf_base_link_to_map.transform.translation.x-(-13.64))*(1/.05))
                distance = np.sqrt( (row-y)**2 + (col-x)**2 )
                angle = np.arctan2(y-row, x-col)
                lidar_idx = int((angle-theta) * len(lidar_msg.ranges) / (2*np.pi))
                myrange = lidar_msg.ranges[lidar_idx]/.05
                if distance < myrange - 2:
                    self.neg[y, x] += 1.0
                elif distance < myrange + 2:
                    self.pos[y, x] += 1.0
        mymap = stats.beta.mean(self.neg, self.pos)*100.0
        my_map_msg.data = mymap.reshape(-1).astype(np.int32).tolist()
        self.map_publisher.publish(my_map_msg)



rclpy.init()
mapping_node = MappingNode()
rclpy.spin(mapping_node)
mapping_node.destroy_node()
rclpy.shutdown()

