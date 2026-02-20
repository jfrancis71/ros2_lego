import math
import numpy as np
import rclpy
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

    def process_lidar(self, lidar_msg, tf_base_link_to_map):
        my_map_msg = OccupancyGrid()
        my_map_msg.header = lidar_msg.header
        my_map_msg.header.frame_id = "map"
        my_map_msg.info.resolution = .05
        my_map_msg.info.width = 317
        my_map_msg.info.height = 118
        my_map_msg.info.origin.position.x = -13.640
        my_map_msg.info.origin.position.y = -3.959
        dat = np.zeros([118, 317]) + self.prior_occupancy_prob
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
                    dat[y, x] = 1.0
                elif distance < myrange + 2:
                    dat[y, x] = 100.0


        my_map_msg.data = dat.reshape(-1).astype(np.int32).tolist()
        self.map_publisher.publish(my_map_msg)



rclpy.init()
mapping_node = MappingNode()
rclpy.spin(mapping_node)
mapping_node.destroy_node()
rclpy.shutdown()

