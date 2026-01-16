import time
import copy
import itertools
import os
import yaml
import numpy as np
import scipy
from scipy.linalg import circulant
from scipy import ndimage as ndi
from scipy.stats import bernoulli, uniform, norm, vonmises
import skimage
from skimage._shared.utils import _to_ndimage_mode
from skimage._shared.utils import convert_to_float
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Header 
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped, PoseWithCovarianceStamped
from tf2_ros.transform_listener import TransformListener
# Current StaticTransformBroadcaster is broken, we need to use from rolling.
# clone git clone https://github.com/ros2/geometry2.git
# Prepend ./src/geometry2/tf2_ros_py/tf2_ros to PYTHONPATH and export
from static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster, TransformException
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros.buffer import Buffer


def angle_diff(angle_1, angle_2):
    """Returns closest difference between two angles.

       Note you cannot just subtract the angles, eg 3.1 - (-3.1) = 6.2. This not the closest angle change.

    """
    abs_diff_angle = np.abs(angle_1 - angle_2)
    return np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))


class MCL:
    def __init__(self, map_image, origin, resolution):
        self.map_image = (map_image == 0)  # binarize the image
        self.origin = origin
        self.resolution = resolution
        self.num_particles = 500
        self.num_angles = 360  # Number of buckets in our angle quantization
        self.max_radius = 100  # Maximum radius in pixels that we make predictions over.
        # below is magic number to compensate lidar scan points not independent.
        self.correlation_log_prob_factor = 100.0
        self.map_image_height = map_image.shape[0]
        self.map_width = map_image.shape[1] * self.resolution
        self.map_height = map_image.shape[0] * self.resolution
        self.particles = None #  Particles are with respect to laser orientation
        k_radius = 100 / 100
        k_angle = self.num_angles / (2 * np.pi)
        def polar_map_fn(output_coords):
            angle = output_coords[:, 1] / k_angle
            rr = ((output_coords[:, 0] / k_radius) * np.sin(angle))
            cc = ((output_coords[:, 0] / k_radius) * np.cos(angle))
            coords = np.column_stack((cc, rr))
            return coords
        polar_map = skimage.transform.warp_coords(polar_map_fn, (self.num_angles, self.max_radius))
        # The last column gives (x,y) coordinates in our image grid of point
        # using polar coordinates from the first two indices.
        # polar_map has shape [self.num_angles, self.num_radius, 1, 2]
        # The dim 1 in 2nd from last above is for fast broadcast with particles
        self.polar_map = np.transpose(polar_map, axes=(1, 2, 0))[:, :, np.newaxis, :]

    def initial_pose(self, x, y, angle):
        self.particles = np.tile(
            np.array([x, y, angle]), reps=(self.num_particles, 1))

    def expected_pose(self):
        x_mean, y_mean, _ = np.mean(self.particles[:self.num_particles], axis=0)
        _, angle, _ = vonmises.fit(self.particles[:self.num_particles, 2], fscale=1)
        return x_mean, y_mean, angle

    def update_particles_odom(self, previous_odom_pose, current_odom_pose):
        # p.136 Probabilistic Robotics
        # My angle error model is slightly different from above.
        # Book assumes robot rotates to head in direction in which it actually
        # travelled.
        # Below model is more suitable for holonomic robot, or where lidar is
        # mounted in different direction to robot direction of travel.
        alpha1 = 0.15
        alpha3 = 0.05
        diff_x = current_odom_pose[0] - previous_odom_pose[0]
        diff_y = current_odom_pose[1] - previous_odom_pose[1]
        d_rot1 = np.arctan2(diff_y, diff_x) - previous_odom_pose[2]
        d_trans = np.sqrt(diff_y**2 + diff_x**2)
        d_rot2 = current_odom_pose[2] - previous_odom_pose[2] - d_rot1
        diff_angle = angle_diff(current_odom_pose[2], previous_odom_pose[2])
        sample_d_rot1 = d_rot1 + np.random.normal(size=self.num_particles)*diff_angle*alpha1
        sample_d_trans = d_trans + np.random.normal(size=self.num_particles)*d_trans*alpha3
        sample_d_rot2 = d_rot2 + np.random.normal(size=self.num_particles)*diff_angle*alpha1
        self.particles[:, 0] += sample_d_trans * np.cos(self.particles[:, 2] + sample_d_rot1)
        self.particles[:, 1] += sample_d_trans * np.sin(self.particles[:, 2] + sample_d_rot1)
        self.particles[:, 2] += sample_d_rot1 + sample_d_rot2

    def update_particles_lidar(self, scan):
        new_scan = skimage.transform.resize(scan.astype(np.float32), (self.num_angles,))
        predictions = self.range_predictions(self.particles)
        logprobs_ranges = self.logprob_range_predictions(predictions, new_scan[np.newaxis, :])
        logprobs_particles = logprobs_ranges.sum(axis=1)/self.correlation_log_prob_factor
        probs = np.exp(logprobs_particles)
        norm_probs = probs/probs.sum()
        self.particles = self.resample_particles(self.particles, norm_probs)

    def logprob_range_predictions(self, predictions, scan_line):
        z_hit, z_rand = .99, .01
        log_z_hit, log_z_rand = np.log(z_hit), np.log(z_rand)
        log_in_range_density = norm.logpdf(scan_line, loc=predictions, scale=.1)
        log_out_of_range_density = uniform.logpdf(scan_line, loc=np.zeros_like(predictions) + 5.0, scale=20.0)
        in_range_indices = np.where(predictions>=-.5)
        out_of_range_indices = np.where(predictions<-.5)
        log_prediction_density = np.zeros_like(predictions)
        log_prediction_density[in_range_indices] = log_in_range_density[in_range_indices]
        log_prediction_density[out_of_range_indices] = log_out_of_range_density[out_of_range_indices]
        log_rand_density = uniform.logpdf(scan_line, loc=predictions*0.0, scale=25.0)
        log_density = np.nan_to_num(scipy.special.logsumexp(np.stack([log_prediction_density + log_z_hit, log_rand_density + log_z_rand]), axis=0))
        logpdf = np.nan_to_num(log_density)
        return log_density

    def range_predictions(self, particles):
        image_coord = np.zeros_like(particles)
        image_coord[:, 0] = self.map_image_height - (particles[:, 1] - self.origin[1])/self.resolution
        image_coord[:, 1] = (particles[:, 0] - self.origin[0])/self.resolution
        map_coords = np.transpose(self.polar_map + image_coord[:, :2], axes=(3,2,0,1))
        polar_coord_predictions = ndi.map_coordinates(self.map_image, map_coords, prefilter=False, order=0, cval=0.0)
        skimage.transform._warps._clip_warp_output(self.map_image, polar_coord_predictions, 'constant', 0.0, True)
        # where argmax has several equal maxima, it returns the first.
        # hence returns closest point
        polar_coords = np.argmax(polar_coord_predictions, axis=2)*self.resolution
        out_of_range_indices = np.where(np.max(polar_coord_predictions, axis=2)==0)
        polar_coords[out_of_range_indices] = -1
        predictions = np.array([ np.flip(np.roll(polar_coords[particle_id], int(360 * particles[particle_id, 2] / (2 * np.pi)))) for particle_id in range(len(particles))])
        return predictions

    def resample_particles(self, particles, probs):
        resampled_particle_indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=probs)
        resampled_particles = particles[resampled_particle_indices]
        return resampled_particles


class MCLNode(Node):
    def __init__(self):
        super().__init__("mcl_node")
        self.declare_parameter('map', 'my_house.yaml')
        self.map_file = self.get_parameter('map').get_parameter_value().string_value
        with open(self.map_file, 'r') as map_file:
            map_properties = yaml.safe_load(map_file)
            image_filename = map_properties['image']
            origin = map_properties['origin']
            resolution = map_properties["resolution"]
        map = skimage.io.imread(os.path.join(os.path.split(self.map_file)[0], image_filename))
        self.mcl = MCL(map, origin, resolution)
        self.initial_pose_received = False
        self.min_dist = 0.03  # minimum distance for lidar update
        self.min_angle = .08  # minimum angle change for lidar update
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.initialpose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.initialpose_callback,
            1)
        self.pred_publisher = \
            self.create_publisher(LaserScan, "/pred_laser", 1)
        self.pdf_publisher = \
            self.create_publisher(LaserScan, "/pdf", 1)
        self.marker_pdf_publisher = self.create_publisher(Marker, '/particles_marker', 1)
        lidar_msg = LaserScan()
        lidar_msg.angle_min = 0.0
        lidar_msg.angle_max = 2 * np.pi
        lidar_msg.angle_increment = 2 * np.pi / 360
        lidar_msg.time_increment = 0.00019850002718158066
        lidar_msg.scan_time = 0.10004401206970215
        lidar_msg.range_min = 0.019999999552965164
        lidar_msg.range_max = 25.0
        lidar_msg.header.frame_id = "base_laser"
        self.template_lidar_msg = lidar_msg
        self.tf_buffer = Buffer()
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False, qos=qos)
        self.current_lidar_msg = None
        self.previous_odom_pose = None
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

    def publish_map_odom_transform(self, base_laser_to_odom_tf, pose):
        zero_to_odom_tf = TransformStamped()
        zero_to_odom_tf.header.stamp = self.get_clock().now().to_msg()
        zero_to_odom_tf.header.frame_id = 'zero'
        zero_to_odom_tf.child_frame_id = 'odom'
        zero_to_odom_tf.transform = base_laser_to_odom_tf.transform
        map_to_zero_tf = TransformStamped()
        map_to_zero_tf.header.stamp = self.get_clock().now().to_msg()
        map_to_zero_tf.header.frame_id = 'map'
        map_to_zero_tf.child_frame_id = 'zero'
        m_to_z_tf_trans = map_to_zero_tf.transform.translation
        m_to_z_tf_trans.x, m_to_z_tf_trans.y, m_to_z_tf_trans.z = pose[0], pose[1], 0.0
        q = quaternion_from_euler(0, 0, pose[2])
        m_to_z_tf_rot = map_to_zero_tf.transform.rotation
        m_to_z_tf_rot.x, m_to_z_tf_rot.y, m_to_z_tf_rot.z, m_to_z_tf_rot.w = q[0], q[1], q[2], q[3]
        self.tf_static_broadcaster.sendTransform([zero_to_odom_tf, map_to_zero_tf])

    def publish_lidar_prediction(self, stamp, ranges):
        lidar_msg = copy.deepcopy(self.template_lidar_msg)
        lidar_msg.ranges = ranges
        lidar_msg.header.stamp = stamp
        self.pred_publisher.publish(lidar_msg)

    def publish_pdf(self, stamp, pdf):
        lidar_msg = copy.deepcopy(self.template_lidar_msg)
        squashed_pdf = -np.tanh(pdf)+2
        lidar_msg.ranges = squashed_pdf
        lidar_msg.header.stamp = stamp
        self.pdf_publisher.publish(lidar_msg)

    def publish_particles(self, stamp, pose):
        marker = Marker()
        marker.header.stamp = stamp
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
        location = self.mcl.particles[:, :2] - pose[:2]
        rot_points = np.zeros([self.mcl.num_particles, 2])
        rot_points[:, 0] = location[:, 0] * np.cos(-pose[2]) - location[:, 1] * np.sin(-pose[2])
        rot_points[:, 1] = location[:, 0] * np.sin(-pose[2]) + location[:, 1] * np.cos(-pose[2])
        marker.points = [Point(x=x,y=y) for (x, y) in rot_points.tolist()]
        marker.frame_locked = True
        self.marker_pdf_publisher.publish(marker)

    def ros2_to_pose(self, odom_transform):
        trans_tf = odom_transform.transform.translation
        rot_tf = odom_transform.transform.rotation
        rot = [rot_tf.x, rot_tf.y, rot_tf.z, rot_tf.w]
        _, _, theta = euler_from_quaternion(rot)
        return (trans_tf.x, trans_tf.y, theta)

    def publish_ros2(self, header, base_link_to_odom_transform, pose, predictions, log_prob):
        self.publish_map_odom_transform(base_link_to_odom_transform, pose)
        self.publish_lidar_prediction(header.stamp, predictions)
        self.publish_pdf(header.stamp, log_prob)
        self.publish_particles(header.stamp, pose)

    def initialpose_callback(self, initialpose_msg):
        try:
            base_link_to_base_laser_transform = self.tf_buffer.lookup_transform(
                "base_link",
                "base_laser",
                rclpy.time.Time())
        except TransformException as ex:
            print("No Transform for base_link to base_laser, initial pose not set.")
            return
        self.initial_pose_received = True
        r_t = initialpose_msg.pose.pose.orientation
        rot = [r_t.x, r_t.y, r_t.z, r_t.w]
        _, _, theta_laser = euler_from_quaternion(rot)
        # I am assuming here that laser and base_link just differ by orientation
        _, _, theta_diff = self.ros2_to_pose(base_link_to_base_laser_transform)
        self.mcl.initial_pose(initialpose_msg.pose.pose.position.x, initialpose_msg.pose.pose.position.y, theta_laser + theta_diff)
        print("Initial pose set.")

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
            tf_base_laser_to_odom = self.tf_buffer.lookup_transform(
                "base_laser",
                "odom",
                lidar_msg_time)  # https://github.com/ros2/ros2_documentation/issues/4385
            tf_odom_to_base_laser = self.tf_buffer.lookup_transform(
                "odom",
                "base_laser",
                lidar_msg_time)
        except TransformException as ex:
            print("Transform exception: ", ex)
            return
        self.process_lidar(self.current_lidar_msg, tf_base_laser_to_odom, tf_odom_to_base_laser)
        self.current_lidar_msg = None

    def process_lidar(self, lidar_msg, tf_base_laser_to_odom, tf_odom_to_base_laser):
        scan = np.array(lidar_msg.ranges)
        current_odom_pose = self.ros2_to_pose(tf_odom_to_base_laser)
        if self.previous_odom_pose is None:
            self.previous_odom_pose = current_odom_pose
        if self.robot_moved(current_odom_pose):
            self.mcl.update_particles_odom(self.previous_odom_pose, current_odom_pose)
            self.mcl.update_particles_lidar(scan)
            self.previous_odom_pose = current_odom_pose
        pose = self.mcl.expected_pose()
        mean_predictions = self.mcl.range_predictions(np.array([[pose[0], pose[1], pose[2]]]))
        new_scan = skimage.transform.resize(scan.astype(np.float32), (self.mcl.num_angles,))
        logprob_ranges = self.mcl.logprob_range_predictions(mean_predictions, new_scan[np.newaxis, :])
        self.publish_ros2(lidar_msg.header, tf_base_laser_to_odom, pose, mean_predictions[0], logprob_ranges[0])


rclpy.init()
mcl_node = MCLNode()
rclpy.spin(mcl_node)
mcl_node.destroy_node()
rclpy.shutdown()
