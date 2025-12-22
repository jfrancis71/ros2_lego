import time
import copy
import itertools
import os
import yaml
import skimage
import numpy as np
from scipy.linalg import circulant
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header 
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
# Current StaticTransformBroadcaster is broken, we need to use from rolling.
# clone git clone https://github.com/ros2/geometry2.git
# Prepend ./src/geometry2/tf2_ros_py/tf2_ros to PYTHONPATH and export
from static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler
from tf_transformations import euler_from_quaternion
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from tf2_ros import TransformException
from rclpy.time import Time
from scipy import ndimage as ndi
from skimage._shared.utils import _to_ndimage_mode
from skimage._shared.utils import convert_to_float
from scipy.stats import vonmises
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import bernoulli
import scipy


class MCL:
    def __init__(self, map_image, origin, resolution):
        self.map_image = (map_image == 0)
        self.origin = origin
        self.resolution = resolution
        self.num_particles = 500
        self.replacement = 400
        self.num_angles = 360  # Number of buckets in our angle quantization
        self.max_radius = 100  # Maximum radius in pixels that we make predictions over.
        self.map_image_height = map_image.shape[0]
        self.map_width = map_image.shape[1] * self.resolution
        self.map_height = map_image.shape[0] * self.resolution
        self.particles = np.transpose(np.array([ self.origin[0] + self.map_width*np.random.random(size=self.num_particles), self.origin[1] + self.map_height*np.random.random(size=self.num_particles), 2 * np.pi * np.random.random(size=self.num_particles) ]))
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

    def init(self, x, y, angle):
        self.particles[:, 0] = x
        self.particles[:, 1] = y
        self.particles[:, 2] = angle

    def update_motion_particles(self, old_odom_pose, new_odom_pose):
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

    def predictions(self, particles):
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

    def prediction_prob(self, predictions, scan_line):
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
        logs = logpdf.sum(axis=1)
        print("BEST=", logs.max())
        return logpdf, logs

    def expected_pose(self, particles):
        x_mean, y_mean, _ = np.mean(particles, axis=0)
        y_std, x_std, _ = np.std(particles, axis=0)
        kappa, angle, _ = vonmises.fit(particles[:, 2], fscale=1)
        angle_std = 1/np.sqrt(kappa)
        return (x_mean, y_mean, angle), (x_std, y_std, angle_std)

    def update_lidar_particles(self, scan):
        new_scan = skimage.transform.resize(scan.astype(np.float32), (self.num_angles,))
        new_scan = np.roll(new_scan, -90)  # account for laser mounting.
        predictions = self.predictions(self.particles)
        _, logprobs = self.prediction_prob(predictions, new_scan[np.newaxis, :])
        logprobs = logprobs/100
        probs = np.exp(logprobs)
        probs = probs/probs.sum()
        self.particles = self.resample_particles(self.particles, probs)


class LocalizerNode(Node):
    def __init__(self):
        super().__init__("localizer")
        self.declare_parameter('map', 'my_house.yaml')
        self.map_file = self.get_parameter('map').get_parameter_value().string_value
        with open(self.map_file, 'r') as map_file:
            map_properties = yaml.safe_load(map_file)
            image_filename = map_properties['image']
            origin = map_properties['origin']
            resolution = map_properties["resolution"]
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
        self.particles_resampled_publisher = \
            self.create_publisher(PointCloud2, "/metropolis_particles", 1)
        self.marker_loc_uncertainty_publisher = self.create_publisher(Marker, 'loc_uncertainty', 1)
        self.marker_pdf_publisher = self.create_publisher(Marker, '/particles_marker', 1)
        self.angle_uncertainty_publisher = self.create_publisher(Marker, 'angle_uncertainty', 1)
        self.tf_buffer = Buffer()
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True, qos=qos)
        map = skimage.io.imread(os.path.join(os.path.split(self.map_file)[0], image_filename))
        self.localizer = MCL(map, origin, resolution)
        self.old_transform = None
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.init_phase = 0
        self.diff_t = 0.03
        self.diff_angle = .08

    def send_map_base_link_transform(self, base_link_to_odom_tf, pose):
        try:
            base_laser_to_base_link_tf = self.tf_buffer.lookup_transform(
                "base_laser",
                "base_link",
                rclpy.time.Time())
        except TransformException as ex:
            print("No Transform")
            return
        zero_to_odom_tf = TransformStamped()
        zero_to_odom_tf.header.stamp = self.get_clock().now().to_msg()
        zero_to_odom_tf.header.frame_id = 'zero'
        zero_to_odom_tf.child_frame_id = 'odom'
        zero_to_odom_tf.transform = base_link_to_odom_tf.transform
        map_to_zero_tf = TransformStamped()
        map_to_zero_tf.header.stamp = \
            self.get_clock().now().to_msg()
        map_to_zero_tf.header.frame_id = 'map'
        map_to_zero_tf.child_frame_id = 'zero'
        map_to_zero_tf.transform.translation.x = \
            pose[0]
        map_to_zero_tf.transform.translation.y = \
            pose[1]
        map_to_zero_tf.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, pose[2])
        map_to_zero_tf.transform.rotation.x = q[0]
        map_to_zero_tf.transform.rotation.y = q[1]
        map_to_zero_tf.transform.rotation.z = q[2]
        map_to_zero_tf.transform.rotation.w = q[3]
        self.tf_static_broadcaster.sendTransform([zero_to_odom_tf, map_to_zero_tf])

    def publish_lidar_prediction(self, header, ranges):
        lidar_msg1 = LaserScan()
        lidar_msg1.ranges = np.roll(ranges, 90)  # Account for laser mounting
        lidar_msg1.angle_min = 0.0
        lidar_msg1.angle_max = 6.28318548
        lidar_msg1.angle_increment = 2 * np.pi / 360
        lidar_msg1.time_increment = 0.00019850002718158066
        lidar_msg1.scan_time = 0.10004401206970215
        lidar_msg1.range_min = 0.019999999552965164
        lidar_msg1.range_max = 25.0
        lidar_msg1.header = header
        self.pred_publisher.publish(lidar_msg1)

    def publish_pdf(self, header, pdf):
        lidar_msg1 = LaserScan()
        remap = -np.tanh(pdf)+2
        lidar_msg1.ranges = np.roll(remap, 90)  # Account for laser mounting
        lidar_msg1.angle_min = 0.0
        lidar_msg1.angle_max = 6.28318548
        lidar_msg1.angle_increment = 2 * np.pi / 360
        lidar_msg1.time_increment = 0.00019850002718158066
        lidar_msg1.scan_time = 0.10004401206970215
        lidar_msg1.range_min = 0.019999999552965164
        lidar_msg1.range_max = 25.0
        lidar_msg1.header = header
        self.pdf_publisher.publish(lidar_msg1)

    def publish_point_cloud(self, header, pose, particles):
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
        marker.header.stamp = header.stamp
        marker.header.frame_id = "base_link"
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
            p.x = rot_points[l, 0].item()
            p.y = rot_points[l, 1].item()
            p.z = 0.0
            points_list.append(p)
        marker.points = points_list
        marker.frame_locked = True
        self.marker_pdf_publisher.publish(marker)

    def publish_resamples_point_cloud(self, header, particles):
        points = np.zeros([self.localizer.replacement, 3])
        points[:, 0] = particles[:self.localizer.replacement, 1]*self.localizer.resolution + self.localizer.origin[0]
        points[:, 1] = (self.localizer.map_height-particles[:self.localizer.replacement, 0])*self.localizer.resolution + self.localizer.origin[1]
        cloud_msg_header = Header()
        cloud_msg_header.stamp = header.stamp
        cloud_msg_header.frame_id = "map"
        cloud_msg = point_cloud2.create_cloud_xyz32(cloud_msg_header, points)
        self.particles_resampled_publisher.publish(cloud_msg)

    def publish_loc_uncertainty_marker(self, header, pose, pose_uncertainty):
        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = "base_link"
        marker.ns = "basic_shapes"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = 0.0, 0.0, 0.0
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = 0.0, 0.0, 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = pose_uncertainty[0], pose_uncertainty[1], 0.5
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.0, 1.0, .2
        marker.frame_locked = True
        self.marker_loc_uncertainty_publisher.publish(marker)

    def publish_angle_uncertainty_marker(self, header, pose_uncertainty):
        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = "base_link"
        marker.ns = "basic"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x, marker.scale.y, marker.scale.z = .1, .1, .1
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.0, 1.0, 1.0
        mstd_angle = np.min((pose_uncertainty[2], np.pi))
        point1 = Point()
        point1.x, point1.y, point1.z = 0.0, 0.0, 0.1
        point2 = Point()
        point2.x, point2.y, point2.z = 0.0 + .5*np.cos(-mstd_angle), 0.0 + .5*np.sin(-mstd_angle), 0.1
        point3 = Point()
        point3.x, point3.y, point3.z = 0.0 + .5*np.cos(+mstd_angle), 0.0 + .5*np.sin(+mstd_angle), 0.1
        marker.points = [point1, point2, point1, point3]
        marker.frame_locked = True
        self.angle_uncertainty_publisher.publish(marker)

    def ros2_to_pose(self, odom_transform):
        t = odom_transform.transform.translation
        r_t = odom_transform.transform.rotation
        rot = [r_t.x, r_t.y, r_t.z, r_t.w]
        _, _, theta = euler_from_quaternion(rot)
        return (t.x, t.y, theta)

    def publish_ros2(self, header, base_link_to_odom_transform, pose, pose_uncertainty, particles, predictions, log_prob):
        self.send_map_base_link_transform(base_link_to_odom_transform, pose)
        self.publish_lidar_prediction(header, predictions)
        self.publish_pdf(header, log_prob)
        self.publish_point_cloud(header, pose, particles)
        self.publish_loc_uncertainty_marker(header, pose, pose_uncertainty)
        self.publish_angle_uncertainty_marker(header, pose_uncertainty)

    def initialpose_callback(self, initialpose_msg):
        self.init_phase = 1
        r_t = initialpose_msg.pose.pose.orientation
        rot = [r_t.x, r_t.y, r_t.z, r_t.w]
        _, _, theta = euler_from_quaternion(rot)
        self.localizer.init(initialpose_msg.pose.pose.position.x, initialpose_msg.pose.pose.position.y, theta)

    def lidar_callback(self, lidar_msg):
        scan = np.array(lidar_msg.ranges)
        try:
            base_link_to_odom_transform = self.tf_buffer.lookup_transform(
                "base_link",
                "odom",
                rclpy.time.Time())
        except TransformException as ex:
            print("No Transform")
            return
        try:
            odom_to_base_link_transform = self.tf_buffer.lookup_transform(
                "odom",
                "base_link",
                rclpy.time.Time())
        except TransformException as ex:
            print("No Transform")
            return
        if self.old_transform is None:
            self.old_transform = odom_to_base_link_transform
        lidar_msg_time = Time.from_msg(lidar_msg.header.stamp)
        odom_base_tf_time = Time.from_msg(odom_to_base_link_transform.header.stamp)
        delay = (lidar_msg_time-odom_base_tf_time).nanoseconds*1e-9
        if delay > .1:
            print("DELAY ", delay)
            return
        if self.init_phase == 0:
            return
        old_odom_pose = self.ros2_to_pose(self.old_transform)
        new_odom_pose = self.ros2_to_pose(odom_to_base_link_transform)
        self.localizer.update_motion_particles(old_odom_pose, new_odom_pose)
        diff_x = new_odom_pose[0] - old_odom_pose[0]
        diff_y = new_odom_pose[1] - old_odom_pose[1]
        d_trans = np.sqrt(diff_y**2 + diff_x**2)
        abs_diff_angle = np.abs(new_odom_pose[2] - old_odom_pose[2])
        diff_angle = np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))
        abs_diff = np.abs(diff_angle)
        if abs_diff > self.diff_angle or d_trans > self.diff_t:
            print("Update scan")
            self.localizer.update_lidar_particles(scan)
        pose, pose_uncertainty = self.localizer.expected_pose(self.localizer.particles[:self.localizer.replacement])
        mean_predictions = self.localizer.predictions(np.array([[pose[1], pose[0], pose[2]]]))
        new_scan = skimage.transform.resize(scan.astype(np.float32), (self.localizer.num_angles,))
        new_scan = np.roll(new_scan, -90)  # account for laser mounting.
        logs, log_prob = self.localizer.prediction_prob(mean_predictions, new_scan[np.newaxis, :])
        self.publish_ros2(lidar_msg.header, base_link_to_odom_transform, pose, pose_uncertainty, self.localizer.particles, mean_predictions[0], log_prob)
        self.old_transform = odom_to_base_link_transform


rclpy.init()
localizer_node = LocalizerNode()
rclpy.spin(localizer_node)
localizer_node.destroy_node()
rclpy.shutdown()
