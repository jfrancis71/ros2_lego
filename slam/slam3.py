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


def angle_diff(angle_1, angle_2):
    """Returns closest difference between two angles.

       Note you cannot just subtract the angles, eg 3.1 - (-3.1) = 6.2. This not the closest angle change.

    """
    abs_diff_angle = np.abs(angle_1 - angle_2)
    return np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))


class SLAM:
    def __init__(self):
        self.num_particles = 30
        self.num_angles = 360
        self.particles = np.tile(np.array([0.0, 0.0, -0.5 * np.pi]), reps=(self.num_particles, 1, 1))
        # shape N, T, P where N is particle no, T is time, P is pose shape
        self.probs = np.zeros([self.num_particles])
        self.odom_poses = []
        self.scans = []

    def init(self, raw_scan, current_odom_pose):
        self.scans.append(np.array(skimage.transform.resize(raw_scan.astype(np.float32), (self.num_angles,))))
        self.odom_poses.append(current_odom_pose)

    def update(self, scan, previous_odom_pose, current_odom_pose):
        self.scans.append(np.array(skimage.transform.resize(scan.astype(np.float32), (self.num_angles,))))
        self.odom_poses.append(current_odom_pose)
        particles = self.extend()[0]
        old_particles = self.particles
        self.particles = np.zeros([self.num_particles, old_particles.shape[1]+1, 3])
        self.particles[:, :-1] = old_particles
        self.particles[:,-1] = particles
        predictions = self.laser_pred()
        logprobs_particles = self.laser_probs(predictions)
        probs = np.exp(logprobs_particles)
        norm_probs = probs/probs.sum()
        self.particles[:, -1] = self.resample_particles(self.particles[:,-1], norm_probs)

    def resample_particles(self, particles, probs):
        resampled_particle_indices = np.random.choice(np.arange(self.num_particles),
size=self.num_particles, p=probs)
        resampled_particles = particles[resampled_particle_indices]
        return resampled_particles

    def extend(self):
        alpha1 = 0.15
        alpha1 = 0.25
        alpha3 = 0.05
        t = len(self.odom_poses)-1
        previous_odom_pose = self.odom_poses[t-1]
        current_odom_pose = self.odom_poses[t]
        diff_x = current_odom_pose[0] - previous_odom_pose[0]
        diff_y = current_odom_pose[1] - previous_odom_pose[1]
        d_rot1 = np.arctan2(diff_y, diff_x) - previous_odom_pose[2]
        d_trans = np.sqrt(diff_y**2 + diff_x**2)
        d_rot2 = current_odom_pose[2] - previous_odom_pose[2] - d_rot1
        diff_angle = angle_diff(current_odom_pose[2], previous_odom_pose[2])
        sp1 = np.random.normal(size=self.num_particles)
        sp2 = np.random.normal(size=self.num_particles)
        sp3 = np.random.normal(size=self.num_particles)
        tprobs = norm.logpdf(sp1) + norm.logpdf(sp2) + norm.logpdf(sp3)
        sample_d_rot1 = d_rot1 + sp1*diff_angle *alpha1
        sample_d_trans = d_trans + sp2*d_trans* alpha3
        sample_d_rot2 = d_rot2 + sp3*diff_angle *alpha1
        sample_d_rot1 = 0.0 + sp1
        sample_d_trans = 0.0 + sp2*d_trans* alpha3
        sample_d_rot2 = 0.0 + sp3
        particles = np.zeros([self.num_particles, 3])
        print("PART SHAPE=", self.particles.shape, "T=", t)
#        particles[:, 0] = self.particles[:, t-1, 0] + sample_d_trans * np.cos(self.particles[:, t-1, 2] + sample_d_rot1)
#        particles[:, 1] = self.particles[:, t-1, 1] + sample_d_trans * np.sin(self.particles[:, t-1, 2] + sample_d_rot1)
#        particles[:,  2] = self.particles[:, t-1, 2] + sample_d_rot1 + sample_d_rot2
        particles[:, 0] = self.particles[:, t-1, 0] + sp1*.1
        particles[:, 1] = self.particles[:, t-1, 1] + sp2*.1
        particles[:,  2] = self.particles[:, t-1, 2] + sp3*.5
        return particles, tprobs

    def expected_pose(self):
        x_mean, y_mean, _ = np.mean(self.particles[:, -1], axis=0)
        _, angle, _ = vonmises.fit(self.particles[:, -1, 2], fscale=1)
        return x_mean, y_mean, angle

    def laser_pred(self):
        predictions = np.zeros([self.num_particles, 360])
        for p in range(self.num_particles):
            ranges = [ [] for _ in range(360)]
            for a in range(360):
                if np.isnan(self.scans[0][a]):
                    continue
                x = self.scans[0][a] * np.cos(a*2*np.pi/360 - np.pi/2)
                y = self.scans[0][a] * np.sin(a*2*np.pi/360 - np.pi/2)
                xp = x - self.particles[p, -1, 0]
                yp = y - self.particles[p, -1, 1]
                na = a*2*np.pi/360 + self.particles[p, -1, 2]
                X = xp * np.cos(na) + yp * np.sin(na)
                Y = -xp * np.sin(na) + yp * np.cos(na)
                R = np.sqrt(X*X + Y*Y)
                THETA = np.arctan2(Y, X)
                R = np.sqrt(xp*xp + yp*yp)
                THETA = np.arctan2(yp, xp) - self.particles[p, -1, 2]
                idx = int(THETA*360/(2*np.pi)) % 360
                ranges[idx].append(R)
            for a in range(360):
                if ranges[a] == []:
                    predictions[p, a] = np.nan
                else:
                    ranges[a].sort()
                    predictions[p, a] = ranges[a][0]
        return predictions

    def laser_probs(self, predictions):
        probs = np.zeros([self.num_particles])
        for p in range(self.num_particles):
            probs[p] = np.nansum(norm.logpdf(self.scans[-1], loc=predictions[p], scale=0.1))
        return probs/1000


class SLAMNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.initial_pose_received = False
        self.min_dist = 0.03  # minimum distance for lidar update
        self.min_angle = .08  # minimum angle change for lidar update
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.marker_pdf_publisher = self.create_publisher(Marker, '/particles_marker', 1)
        self.map_publisher = \
            self.create_publisher(OccupancyGrid, "my_map", 1)
        self.loc_map_publisher = \
            self.create_publisher(OccupancyGrid, "loc_map", 1)
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
        self.init_wait = 0
        self.slam = SLAM()

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
#        self.publish_particles(pose)

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
        scan = np.array(lidar_msg.ranges)
        current_odom_pose = self.ros2_to_pose(tf_odom_to_base_laser)
        if self.init_wait == 10:
            self.slam.init(scan, current_odom_pose)
            self.init_wait += 1
            return
        if self.previous_odom_pose is None:
            self.previous_odom_pose = current_odom_pose
        if self.robot_moved(current_odom_pose):
            self.slam.update(np.array(lidar_msg.ranges), self.previous_odom_pose, current_odom_pose)
            self.previous_odom_pose = current_odom_pose
        pose = self.slam.expected_pose()
        self.publish_ros2(tf_base_laser_to_odom, pose)


rclpy.init()
slam_node = SLAMNode()
rclpy.spin(slam_node)
slam_node.destroy_node()
rclpy.shutdown()
