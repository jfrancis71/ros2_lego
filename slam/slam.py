import copy
import os
import yaml
import numpy as np
#import scipy
#from scipy import ndimage as ndi
from scipy.stats import uniform, norm, vonmises
from scipy.spatial.transform import Rotation as R
import scipy.stats as stats
#import skimage
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


def angle_diff(angle_1, angle_2):
    """Returns closest difference between two angles.

       Note you cannot just subtract the angles, eg 3.1 - (-3.1) = 6.2. This not the closest angle change.

    """
    abs_diff_angle = np.abs(angle_1 - angle_2)
    return np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))


def expected_pose(particles):
    x_mean, y_mean, _ = np.mean(particles, axis=0)
    _, angle, _ = vonmises.fit(particles[:, 2], fscale=1)
    return x_mean, y_mean, angle


def update_particles_odom(particles, previous_odom_pose, current_odom_pose):
    # p.136 Probabilistic Robotics
    # My angle error model is slightly different from above.
    # Book assumes robot rotates to head in direction in which it actually
    # travelled.
    # Below model is more suitable for holonomic robot, or where lidar is
    # mounted in different direction to robot direction of travel.
    alpha1 = 0.15
    alpha3 = 0.05
    num_particles = particles.shape[0]
    diff_x = current_odom_pose[0] - previous_odom_pose[0]
    diff_y = current_odom_pose[1] - previous_odom_pose[1]
    d_rot1 = np.arctan2(diff_y, diff_x) - previous_odom_pose[2]
    d_trans = np.sqrt(diff_y**2 + diff_x**2)
    d_rot2 = current_odom_pose[2] - previous_odom_pose[2] - d_rot1
    diff_angle = angle_diff(current_odom_pose[2], previous_odom_pose[2])
    sample_d_rot1 = d_rot1 + np.random.normal(size=num_particles)*diff_angle *alpha1
    sample_d_trans = d_trans + np.random.normal(size=num_particles)*d_trans* alpha3
    sample_d_rot2 = d_rot2 + np.random.normal(size=num_particles)*diff_angle *alpha1
    new_particles = particles.copy()
    new_particles[:, 0] += sample_d_trans * np.cos(particles[:, 2] + sample_d_rot1)
    new_particles[:, 1] += sample_d_trans * np.sin(particles[:, 2] + sample_d_rot1)
    new_particles[:, 2] += sample_d_rot1 + sample_d_rot2
    return new_particles


class SLAMNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.initial_pose_received = False
        self.min_dist = 0.03  # minimum distance for lidar update
        self.min_angle = .08  # minimum angle change for lidar update
        self.num_particles = 4
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.marker_pdf_publisher = self.create_publisher(Marker, '/particles_marker', 1)
        self.map_publisher = \
            self.create_publisher(OccupancyGrid, "my_map", 1)
        self.tf_buffer = Buffer()
        qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True, qos=qos)
        self.current_lidar_msg = None
        self.previous_odom_pose = None
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.particles = np.tile(np.array([0.0, 0.0, 1.5 * np.pi]), reps=(self.num_particles, 1, 1))
        self.particles = np.tile(np.array([0.0, 0.0, -.5 * np.pi]), reps=(self.num_particles, 1, 1))
        # shape N, T, P where N is particle no, T is time, P is pose shape
        self.neg= np.zeros([self.num_particles, 118, 317]) + 2
        self.pos= np.zeros([self.num_particles, 118, 317]) + 0.5
        self.init_wait = 0

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

    def publish_particles(self, pose):
        marker = Marker()
        marker.header.stamp = self.current_lidar_msg.header.stamp
        marker.header.frame_id = "base_laser"
        marker.ns = "basic_shapes"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = 0.0, 0.0, 0.0
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = 0.0, 0.0, 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = 0.03, 0.03, 0.05
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.3, 1.0, 1.0, .2
        particles_base_laser = np.matmul(self.particles[:, -1, :2] - pose[:2], R.from_rotvec([0, 0, -pose[2]]).as_matrix()[:2, :2])
        marker.points = [Point(x=x,y=y) for (x, y) in particles_base_laser.tolist()]
        marker.frame_locked = True
        self.marker_pdf_publisher.publish(marker)

    def ros2_to_pose(self, odom_transform):
        trans_tf = odom_transform.transform.translation
        rot_tf = odom_transform.transform.rotation
        rot = [rot_tf.x, rot_tf.y, rot_tf.z, rot_tf.w]
        _, _, theta = euler_from_quaternion(rot)
        return (trans_tf.x, trans_tf.y, theta)

    def publish_ros2(self, tf_base_laser_to_odom, pose):
        self.publish_map_odom_transform(tf_base_laser_to_odom, pose)
        self.publish_particles(pose)

    def robot_moved(self, current_odom_pose):
        diff_x = current_odom_pose[0] - self.previous_odom_pose[0]
        diff_y = current_odom_pose[1] - self.previous_odom_pose[1]
        d_trans = np.sqrt(diff_y**2 + diff_x**2)
        diff_angle = angle_diff(current_odom_pose[2], self.previous_odom_pose[2])
        return np.abs(diff_angle) > self.min_angle or d_trans > self.min_dist

    def lidar_callback(self, lidar_msg):
        if self.init_wait < 10:
            if self.init_wait == 0:
                self.process_map(lidar_msg)
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
        scan = np.array(lidar_msg.ranges)
        current_odom_pose = self.ros2_to_pose(tf_odom_to_base_laser)
        if self.previous_odom_pose is None:
            self.previous_odom_pose = current_odom_pose
        if self.robot_moved(current_odom_pose):
            new_particles = update_particles_odom(self.particles[:, -1], self.previous_odom_pose, current_odom_pose)
            self.particles = np.append(self.particles, np.reshape(new_particles, (self.num_particles, 1, 3)), axis=1)
            #new_particles = update_particles_lidar(scan)
            self.previous_odom_pose = current_odom_pose
            print("PROC2")
            self.process_map(lidar_msg)
        pose = expected_pose(self.particles[:, -1])
#        logprob_ranges = self.mcl.logprob_range_predictions(scan)
        self.publish_ros2(tf_base_laser_to_odom, pose)
        lidar_msg_time = Time.from_msg(lidar_msg.header.stamp)
        self.publish_map(lidar_msg)

    def process_map(self, lidar_msg):
        for particle in range(self.num_particles):
            theta = self.particles[particle, -1, 2] - np.pi/2
            theta = self.particles[particle, -1, 2]
            for x in range(317):
                for y in range(118):
                    py = self.particles[particle, -1, 1]
                    px = self.particles[particle, -1, 0]
                    row = int((py-(-3.959))*(1/.05))
                    col = int((px-(-13.64))*(1/.05))
                    distance = np.sqrt( (row-y)**2 + (col-x)**2 )
                    angle = np.arctan2(y-row, x-col)
                    lidar_idx = int((angle-theta) * len(lidar_msg.ranges) / (2*np.pi))
                    print("LID", theta, angle, lidar_idx)
                    myrange = lidar_msg.ranges[lidar_idx]/.05
                    if distance < myrange - 2:
                        self.neg[particle, y, x] += 1.0
                    elif distance < myrange + 2:
                        self.pos[particle, y, x] += 1.0


    def publish_map(self, lidar_msg):
        my_map_msg = OccupancyGrid()
        my_map_msg.header = lidar_msg.header
        my_map_msg.header.frame_id = "map"
        my_map_msg.info.resolution = .05
        my_map_msg.info.width = 317
        my_map_msg.info.height = 118
        my_map_msg.info.origin.position.x = -13.640
        my_map_msg.info.origin.position.y = -3.959
        mymap = stats.beta.mean(self.neg[-1], self.pos[-1])*100.0
        my_map_msg.data = mymap.reshape(-1).astype(np.int32).tolist()
        self.map_publisher.publish(my_map_msg)
        print("Map published")


rclpy.init()
slam_node = SLAMNode()
rclpy.spin(slam_node)
slam_node.destroy_node()
rclpy.shutdown()
