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
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros.buffer import Buffer


class MCL:
    def __init__(self, map_image, origin, resolution):
        self.map_image = (map_image == 0)  # binarize the image
        self.origin = origin
        self.resolution = resolution
        self.num_particles = 500
        self.replacement = 400
        self.num_angles = 360  # Number of buckets in our angle quantization
        self.max_radius = 100  # Maximum radius in pixels that we make predictions over.
        self.map_image_height = map_image.shape[0]
        self.map_width = map_image.shape[1] * self.resolution
        self.map_height = map_image.shape[0] * self.resolution
        self.particles = None #  Particles are with respect to laser orientation
        height = self.num_angles
        k_radius = 100 / 100
        k_angle = height / (2 * np.pi)
        def coord_map_fn(output_coords):
            angle = output_coords[:, 1] / k_angle
            rr = ((output_coords[:, 0] / k_radius) * np.sin(angle))
            cc = ((output_coords[:, 0] / k_radius) * np.cos(angle))
            coords = np.column_stack((cc, rr))
            return coords
        c = skimage.transform.warp_coords(coord_map_fn, (self.num_angles, self.max_radius))
        # The last column gives (x,y) coordinates in our image grid of point
        # using polar coordinates from the first two indices.
        # coord_map has shape [self.num_angles, self.num_radius, 1, 2]
        self.coord_map = np.transpose(c, axes=(1, 2, 0))[:, :, np.newaxis, :]
        self.ndi_mode = _to_ndimage_mode('constant')

    def initial_pose(self, x, y, angle):
        self.particles = np.zeros([self.num_particles, 3])
        self.particles[:, 0] = x
        self.particles[:, 1] = y
        self.particles[:, 2] = angle

    def expected_pose(self, particles):
        x_mean, y_mean, _ = np.mean(particles, axis=0)
        y_std, x_std, _ = np.std(particles, axis=0)
        kappa, angle, _ = vonmises.fit(particles[:, 2], fscale=1)
        angle_std = 1/np.sqrt(kappa)
        return (x_mean, y_mean, angle), (x_std, y_std, angle_std)

    def update_particles_odom(self, old_odom_pose, new_odom_pose):
        #p.136 Probabilistic Robotics
        alpha1 = 0.15  # this is different from book, ignoring d_rot1
                      # Just using angle diffs, better for holonomic
        alpha3 = 0.05
        diff_x = new_odom_pose[0] - old_odom_pose[0]
        diff_y = new_odom_pose[1] - old_odom_pose[1]
        d_rot1 = np.arctan2(diff_y, diff_x) - old_odom_pose[2]
        d_trans = np.sqrt(diff_y**2 + diff_x**2)
        d_rot2 = new_odom_pose[2] - old_odom_pose[2] - d_rot1
        abs_diff_angle = np.abs(new_odom_pose[2] - old_odom_pose[2])
        diff_angle = np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))
        sample_d_rot1 = d_rot1 + np.random.normal(size=self.num_particles)*diff_angle*alpha1
        sample_d_trans = d_trans + np.random.normal(size=self.num_particles)*d_trans*alpha3
        sample_d_rot2 = d_rot2 + np.random.normal(size=self.num_particles)*diff_angle*alpha1
        self.particles[:, 0] += sample_d_trans * np.cos(self.particles[:, 2] + sample_d_rot1)
        self.particles[:, 1] += sample_d_trans * np.sin(self.particles[:, 2] + sample_d_rot1)
        self.particles[:, 2] += sample_d_rot1 + sample_d_rot2

    def update_particles_lidar(self, scan):
        new_scan = skimage.transform.resize(scan.astype(np.float32), (self.num_angles,))
        predictions = self.range_predictions(self.particles)
        logprobs_ranges = self.logprob_range_prediction(predictions, new_scan[np.newaxis, :])
        logprobs = logprobs_ranges.sum(axis=1)
        logprobs = logprobs/100
        probs = np.exp(logprobs)
        probs = probs/probs.sum()
        self.particles = self.resample_particles(self.particles, probs)

    def logprob_range_prediction(self, predictions, scan_line):
        predictions = predictions.copy()
        pdf = norm.logpdf(scan_line, loc=predictions, scale=.1)
        out_range = uniform.logpdf(scan_line, loc=predictions*0.0 + 5.0, scale=20.0)
        wh = np.where(predictions<-.5)
        pdf[wh] = out_range[wh]
        noise = uniform.logpdf(scan_line, loc=predictions*0.0 + 0.0, scale=25.0) + np.log(.01)
        isnan = np.isnan(scan_line)
        valid = (1-isnan)
        stack = np.stack([pdf, noise])
        new_pdf = scipy.special.logsumexp(stack, axis=0)
        logpdf = np.nan_to_num(new_pdf) - 0 * isnan
        return logpdf

    def range_predictions(self, particles):
        image_coord = np.zeros_like(particles)
        image_coord[:, 0] = self.map_image_height - (particles[:, 1] - self.origin[1])/self.resolution
        image_coord[:, 1] = (particles[:, 0] - self.origin[0])/self.resolution
        trans_coords = np.transpose(self.coord_map +image_coord[:, :2], axes=(3,2,0,1))
        polar_coord_predictions = ndi.map_coordinates(self.map_image, trans_coords, prefilter=False, mode=self.ndi_mode, order=0, cval=0.0)
        skimage.transform._warps._clip_warp_output(self.map_image, polar_coord_predictions, 'constant', 0.0, True)
        polar_coords = np.argmax(polar_coord_predictions, axis=2)*self.resolution
        out_of_range = np.where(np.max(polar_coord_predictions, axis=2)==0)
        polar_coords[out_of_range] = -1
        predictions = np.array([ np.flip(np.roll(polar_coords[particle_id], int(360 * particles[particle_id, 2] / (2 * np.pi)))) for particle_id in range(len(particles))])
        return predictions

    def resample_particles(self, particles, probs):
        new_particles = np.zeros_like(self.particles)
        ls = np.array(np.random.choice(np.arange(len(self.particles)), size=self.replacement, p=probs))
        new_particles[:self.replacement, :2] = particles[ls][:, :2]
        new_particles[:self.replacement, 2] = particles[ls][:, 2]
        kidnap_particles = self.num_particles - self.replacement
        new_particles[self.replacement:] = np.transpose(np.array([ self.map_width*np.random.random(size=kidnap_particles), self.map_height*np.random.random(size=kidnap_particles), 2 * np.pi * np.random.random(size=kidnap_particles) ]))
        return new_particles


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
        self.localizer = MCL(map, origin, resolution)
        self.old_transform = None
        self.last_lidar_update_transform = None
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.initial_pose_received = False
        self.diff_t = 0.03
        self.diff_angle = .08
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
        self.tf_buffer = Buffer()
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True, qos=qos)

    def send_map_base_laser_transform(self, base_laser_to_odom_tf, pose):
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
        lidar_msg1 = LaserScan()
        lidar_msg1.ranges = ranges
        lidar_msg1.angle_min = 0.0
        lidar_msg1.angle_max = 6.28318548
        lidar_msg1.angle_increment = 2 * np.pi / 360
        lidar_msg1.time_increment = 0.00019850002718158066
        lidar_msg1.scan_time = 0.10004401206970215
        lidar_msg1.range_min = 0.019999999552965164
        lidar_msg1.range_max = 25.0
        lidar_msg1.header.stamp = stamp
        lidar_msg1.header.frame_id = "base_laser"
        self.pred_publisher.publish(lidar_msg1)

    def publish_pdf(self, stamp, pdf):
        lidar_msg1 = LaserScan()
        squashed_pdf = -np.tanh(pdf)+2
        lidar_msg1.ranges = squashed_pdf
        lidar_msg1.angle_min = 0.0
        lidar_msg1.angle_max = 6.28318548
        lidar_msg1.angle_increment = 2 * np.pi / 360
        lidar_msg1.time_increment = 0.00019850002718158066
        lidar_msg1.scan_time = 0.10004401206970215
        lidar_msg1.range_min = 0.019999999552965164
        lidar_msg1.range_max = 25.0
        lidar_msg1.header.stamp = stamp
        lidar_msg1.header.frame_id = "base_laser"
        self.pdf_publisher.publish(lidar_msg1)

    def publish_point_cloud(self, stamp, pose, particles):
        map_points = np.zeros([self.localizer.replacement, 3])
        new_pose = np.zeros([3])
        map_points[:, 0] = particles[:self.localizer.replacement, 0]
        map_points[:, 1] = particles[:self.localizer.replacement, 1]
        new_pose[:2] = pose[:2]
        points = map_points - new_pose
        rot_points = np.zeros([400, 3])
        rot_points[:, 0] = points[:, 0] * np.cos(-pose[2]) - points[:, 1] * np.sin(-pose[2])
        rot_points[:, 1] = points[:, 0] * np.sin(-pose[2]) + points[:, 1] * np.cos(-pose[2])
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
        points_list = []
        for l in range(rot_points.shape[0]):
            p = Point()
            p.x, p.y, p.z = rot_points[l, 0].item(), rot_points[l, 1].item(), 0.0
            points_list.append(p)
        marker.points = points_list
        marker.frame_locked = True
        self.marker_pdf_publisher.publish(marker)

    def ros2_to_pose(self, odom_transform):
        trans_tf = odom_transform.transform.translation
        rot_tf = odom_transform.transform.rotation
        rot = [rot_tf.x, rot_tf.y, rot_tf.z, rot_tf.w]
        _, _, theta = euler_from_quaternion(rot)
        return (trans_tf.x, trans_tf.y, theta)

    def publish_ros2(self, header, base_link_to_odom_transform, pose, pose_uncertainty, particles, predictions, log_prob):
        self.send_map_base_laser_transform(base_link_to_odom_transform, pose)
        self.publish_lidar_prediction(header.stamp, predictions)
        self.publish_pdf(header.stamp, log_prob)
        self.publish_point_cloud(header.stamp, pose, particles)

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
        self.localizer.initial_pose(initialpose_msg.pose.pose.position.x, initialpose_msg.pose.pose.position.y, theta_laser + theta_diff)
        print("Initial pose set.")

    def lidar_callback(self, lidar_msg):
        if not self.initial_pose_received:
            return
        scan = np.array(lidar_msg.ranges)
        try:
            base_laser_to_odom_transform = self.tf_buffer.lookup_transform(
                "base_laser",
                "odom",
                rclpy.time.Time())
            odom_to_base_laser_transform = self.tf_buffer.lookup_transform(
                "odom",
                "base_laser",
                rclpy.time.Time())
        except TransformException as ex:
            print("No base laser to odom Transform")
            return
        if self.old_transform is None:
            self.old_transform = odom_to_base_laser_transform
            self.last_lidar_update_transform = odom_to_base_laser_transform
        lidar_msg_time = Time.from_msg(lidar_msg.header.stamp)
        odom_base_tf_time = Time.from_msg(odom_to_base_laser_transform.header.stamp)
        delay = (lidar_msg_time-odom_base_tf_time).nanoseconds*1e-9
        if delay > .1:
            print("DELAY ", delay)
            return
        old_odom_pose = self.ros2_to_pose(self.old_transform)
        new_odom_pose = self.ros2_to_pose(odom_to_base_laser_transform)
        self.localizer.update_particles_odom(old_odom_pose, new_odom_pose)
        last_lidar_update_pose = self.ros2_to_pose(self.last_lidar_update_transform)
        diff_x = new_odom_pose[0] - last_lidar_update_pose[0]
        diff_y = new_odom_pose[1] - last_lidar_update_pose[1]
        d_trans = np.sqrt(diff_y**2 + diff_x**2)
        abs_diff_angle = np.abs(new_odom_pose[2] - last_lidar_update_pose[2])
        diff_angle = np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))
        abs_diff = np.abs(diff_angle)
        if abs_diff > self.diff_angle or d_trans > self.diff_t:
            self.localizer.update_particles_lidar(scan)
            self.last_lidar_update_transform = odom_to_base_laser_transform
        time.sleep(.2)  # This helps with "future transform" error. Why?
        pose, pose_uncertainty = self.localizer.expected_pose(self.localizer.particles[:self.localizer.replacement])
        mean_predictions = self.localizer.range_predictions(np.array([[pose[0], pose[1], pose[2]]]))
        new_scan = skimage.transform.resize(scan.astype(np.float32), (self.localizer.num_angles,))
        logprob_ranges = self.localizer.logprob_range_prediction(mean_predictions, new_scan[np.newaxis, :])
        self.publish_ros2(lidar_msg.header, base_laser_to_odom_transform, pose, pose_uncertainty, self.localizer.particles, mean_predictions[0], logprob_ranges[0])
        self.old_transform = odom_to_base_laser_transform


rclpy.init()
mcl_node = MCLNode()
rclpy.spin(mcl_node)
mcl_node.destroy_node()
rclpy.shutdown()
