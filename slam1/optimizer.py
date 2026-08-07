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
import atexit
import random


def angle_diff(angle_1, angle_2):
    """Returns closest difference between two angles.

       Note you cannot just subtract the angles, eg 3.1 - (-3.1) = 6.2. This not the closest angle change.

    """
    abs_diff_angle = np.abs(angle_1 - angle_2)
    return np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))


class SLAM:
    def __init__(self, filename):
        self.num_particles = 24
        self.num_angles = 360
        self.probs = np.zeros([self.num_particles])
        data = np.load(filename)
        self.odom = data["odom"]
        self.scans = data["scans"]
        self.particles = np.tile(np.array([0.0, 0.0, -0.5 * np.pi]), reps=(self.num_particles, self.odom.shape[0], 1))
        # shape N, T, P where N is particle no, T is time, P is pose shape
        print("Scans=", self.scans)

    def rollout(self):
        self.particles = np.tile(np.array([0.0, 0.0, -0.5 * np.pi]), reps=(self.num_particles, self.odom.shape[0], 1))
        for t in range(1, self.odom.shape[0]):
#            print("odom=", self.odom[i])
            self.particles[:, t] = self.extend(self.particles[:, t-1], self.odom[t-1])
            predictions = self.laser_pred(t)
            logprobs_particles = self.laser_probs(predictions, self.scans[t])
            probs = np.exp(logprobs_particles)
            norm_probs = probs/probs.sum()
            self.particles[:, t] = self.resample_particles(self.particles[:,t], norm_probs)


    def update(self, scan, robot_frame_odom):
        particles = self.extend(robot_frame_odom)
        self.odom_delta.append(robot_frame_odom)

    def resample_particles(self, particles, probs):
        resampled_particle_indices = np.random.choice(np.arange(self.num_particles),
size=self.num_particles, p=probs)
        resampled_particles = particles[resampled_particle_indices]
        return resampled_particles

    def extend(self, last_particles, robot_frame_odom):
        particles = np.zeros([self.num_particles, 3])
        sp1 = np.random.normal(size=self.num_particles)
        sp2 = np.random.normal(size=self.num_particles)
        sp3 = np.random.normal(size=self.num_particles)
        forward = robot_frame_odom[0]
        sample_forward = forward + .1*sp1*forward
        slide = robot_frame_odom[1]
        sample_slide = slide + .1*sp2*slide
        spin = robot_frame_odom[2]
        sample_spin = spin + .1*sp3*spin
        particles[:, 0] = last_particles[:, 0] + sample_forward * np.cos(self.particles[:, -1, 2]) + sample_slide * np.cos(self.particles[:, -1, 2] + np.pi/2)
        particles[:, 1] = last_particles[:, 1] + sample_forward * np.sin(self.particles[:, -1, 2]) + sample_slide * np.sin(self.particles[:, -1, 2] + np.pi/2)
        particles[:, 2] = last_particles[:, 2] + sample_spin
        return particles

    def likelihood(self):
        likelihood = np.zeros([self.num_particles])
        for t in range(1, self.odom.shape[1]):
            predictions = self.laser_pred(t)
            logprobs_particles = self.laser_probs(predictions, self.scans[t])
            likelihood += logprobs_particles
        return likelihood


    def expected_pose(self):
        x_mean, y_mean, _ = np.mean(self.particles[:, -1], axis=0)
        _, angle, _ = vonmises.fit(self.particles[:, -1, 2], fscale=1)
        return x_mean, y_mean, angle

    def laser_pred(self, t):
        predictions = np.zeros([self.num_particles, 360])
        for p in range(self.num_particles):
            rel_node = self.select(p, self.particles[p, t], t)
            predictions[p] = self.pred(self.scans[rel_node], self.particles[p, rel_node], self.particles[p, t])
        return predictions

    def interior(self, scan, scan_pose, query_pose):
        xp = scan_pose[0] - query_pose[0]
        yp = scan_pose[1] - query_pose[1]
        R = np.sqrt(xp*xp + yp*yp)
        if R < .5:
            return 1
        else:
            return 0

    def select(self, particle_idx, particle, pt):  # We discretise pose to 1m and ask for pose closest to this
        min_dist = 10000
        for t in range(pt):
            dist = np.sqrt( (self.particles[particle_idx, t, 0]-int(particle[0]))**2 + (self.particles[particle_idx, t, 1] - int(particle[1]))**2)
            if dist < min_dist:
                min_dist = dist
                idx = t
        return idx


    # Computes range predictions from the initial pose to a query pose
    def pred(self, scan, scan_pose, query_pose):
        ranges = [ [] for _ in range(360)]
        for a in range(360):
            if np.isnan(scan[a]):
                continue
            x = scan_pose[0] + scan[a] * np.cos(a*2*np.pi/360 + scan_pose[2])
            y = scan_pose[1] + scan[a] * np.sin(a*2*np.pi/360 + scan_pose[2])
            xp = x - query_pose[0]
            yp = y - query_pose[1]
            na = a*2*np.pi/360 + query_pose[2]
            X = xp * np.cos(na) + yp * np.sin(na)
            Y = -xp * np.sin(na) + yp * np.cos(na)
            R = np.sqrt(X*X + Y*Y)
            THETA = np.arctan2(Y, X)
            R = np.sqrt(xp*xp + yp*yp)
            THETA = np.arctan2(yp, xp) - query_pose[2]
            idx = int(THETA*360/(2*np.pi)) % 360
            ranges[idx].append(R)
        for a in range(360):
            if ranges[a] == []:
                ranges[a] = np.nan
            else:
                ranges[a].sort()
                ranges[a] = ranges[a][0]
        return ranges

    def laser_probs(self, predictions, scan):
        probs = np.zeros([self.num_particles])
        for p in range(self.num_particles):
            probs[p] = np.nanmean(norm.logpdf(scan, loc=predictions[p], scale=0.1))
        return probs/1000


class SLAMNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.initial_pose_received = False
        self.min_dist = 0.03  # minimum distance for lidar update
        self.min_angle = .08  # minimum angle change for lidar update
        self.marker_pdf_publisher = self.create_publisher(Marker, '/particles_marker', 1)
        self.init_wait = 0
        self.declare_parameter('filename', 'odom.npz')
        filename = self.get_parameter('filename').get_parameter_value().string_value
        self.slam = SLAM(filename)
        self.colors = [ [random.random(), random.random(), random.random()] for i in range(100)]

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
        particles_base_laser = np.matmul(self.slam.particles[:, -1, :2] - pose[:2], R.from_rotvec([0, 0, -pose[2]]).as_matrix()[:2, :2])
        marker.points = [Point(x=x,y=y) for (x, y) in particles_base_laser.tolist()]
        marker.frame_locked = True
        self.marker_pdf_publisher.publish(marker)

    def publish(self, name, id, particle, scan, color):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "d"
        marker.id = id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = 0.0, 0.0, 0.0
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = 0.0, 0.0, 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = 0.03, 0.03, 0.05
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color[0], color[1], color[2], 1.0
        mylist = []
        for b in range(360):
            x = np.cos(b*2*np.pi/360 + particle[2])*scan[b] + particle[0]
            y = np.sin(b*2*np.pi/360 + particle[2])*scan[b] + particle[1]
            if np.isnan(scan[b]):
                pass
            else:
                mylist.append([x,y])

        marker.points = [Point(x=x,y=y) for (x,y) in mylist]
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

    def particle_info(self, particle_idx):
        positions = np.unique(self.slam.particles[particle_idx, :,:2].astype(np.int32), axis=0)
        for p in range(positions.shape[0]):
            ref = self.slam.select(particle_idx, positions[p], self.slam.particles.shape[1])
            print("Pos=", positions[p], ref)
            self.publish(str(p), p, self.slam.particles[particle_idx, ref], self.slam.scans[ref], self.colors[p])

    def run(self):
        print("Hello")
        best_prob = -1000000
        while True:
            self.slam.rollout()
            prob = self.slam.likelihood()
            est_prob = prob.max()
            idx = prob.argmax()
            if est_prob > best_prob:
                best_prob = est_prob
                best_particle = self.slam.particles[idx].copy()
                self.particle_info(idx)
                print(best_particle, best_prob)

    def exit(self):
        print("Exiting")
        self.destroy_node()
#        rclpy.shutdown()



rclpy.init()
slam_node = SLAMNode()
atexit.register(slam_node.exit)
slam_node.run()
