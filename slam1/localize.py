import random
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import slam_utils


class Localizer:
    def __init__(self, filename):
        self.num_particles = 24
        self.particles = np.tile(np.array([0.0, 0.0, -0.5 * np.pi]), reps=(self.num_particles, 3))
        data = np.load(filename)
        self.poses = data["poses"]
        self.scans = data["scans"]
        self.particles = np.random.normal(size=[self.num_particles, 3])

class LocalizerNode(Node):
    def __init__(self):
        super().__init__("localizer_node")
        self.min_dist = 0.03  # minimum distance for lidar update
        self.min_angle = .08  # minimum angle change for lidar update
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.view_publisher = self.create_publisher(Marker, '/view_marker', 1)
        self.view_publisher.publish(Marker())
        return
        self.tf_buffer = Buffer()
        qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True,
qos=qos)
        self.current_lidar_msg = None
        self.previous_odom_pose = None
        self.init_wait = 0
        self.declare_parameter('map_filename', 'map_odom.npz')
        map_filename = self.get_parameter('map_filename').get_parameter_value().string_value
        self.localizer = Localizer(map_filename)
        self.colors = [ [random.random(), random.random(), random.random(), 1.0] for
i in range(100)]
        self.publish_map()
        my = Marker()
        self.view_publisher.publish(my)

    def publish_map(self):
        flat_points = []
        flat_colors = []
        for idx in range(len(self.localizer.poses)):
            print("Pos=", idx)
            points = slam_utils.create_view(self.localizer.poses[idx], self.localizer.scans[idx])
            for point in points:
                flat_points.append(point)
                flat_colors.append(self.colors[idx])
        self.view_publisher.publish(slam_utils.publish_points(self.view_publisher, self.get_clock().now().to_msg(), flat_points, flat_colors))


    def publish_map_odom_transform(self, tf_base_laser_to_odom, pose):
        tf_zero_to_odom = TransformStamped()
        tf_zero_to_odom.header.stamp = self.current_lidar_msg.header.stamp
        tf_zero_to_odom.header.frame_id = 'zero'
        tf_zero_to_odom.child_frame_id = 'odom'
        tf_zero_to_odom.transform = tf_base_laser_to_odom.transform
        tf_map_to_zero = TransformStamped()
        tf_map_to_zero.header.stamp = self.current_lidar_msg.header.stamp
        tf_map_to_zero.header.frame_id = 'map'
        tf_map_to_zero.child_frame_id = 'zero'
        tf_m_to_z_trans = tf_map_to_zero.transform.translation
        tf_m_to_z_trans.x, tf_m_to_z_trans.y, tf_m_to_z_trans.z = pose[0], pose[1], 0.0
        q = quaternion_from_euler(0, 0, pose[2])
        tf_m_to_z_rot = tf_map_to_zero.transform.rotation
        tf_m_to_z_rot.x, tf_m_to_z_rot.y, tf_m_to_z_rot.z, tf_m_to_z_rot.w = q[0], q[1], q[2], q[3]
        self.tf_static_broadcaster.sendTransform([tf_zero_to_odom, tf_map_to_zero])

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
            self.init_wait += 1
            return
        if self.previous_odom_pose is None:
            self.previous_odom_pose = current_odom_pose
        if self.robot_moved(current_odom_pose):
            robot_frame_odom = self.robot_frame_odom(self.previous_odom_pose, current_odom_pose)
            pose = self.localizer.expected_pose()
            self.publish_map_odom_transform(tf_base_laser_to_odom, pose)
            self.previous_odom_pose = current_odom_pose


rclpy.init()
localizer_node = LocalizerNode()
#localizer_node.publish_map()
rclpy.spin(localizer_node)
localizer_node.destroy_node()
rclpy.shutdown()

