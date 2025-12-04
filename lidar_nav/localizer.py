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


class MCL:
    def __init__(self, map_image, origin, resolution):
        self.map_image = map_image
        self.origin = origin
        self.resolution = resolution
        self.map_height = map_image.shape[0]
        self.map_width = map_image.shape[1]
        self.num_particles = 500
        self.replacement = 400
        self.num_angles = 360  # Number of buckets in our angle quantization
        self.max_radius = 100  # Maximum radius in pixels that we make predictions over.
        self.particles = np.transpose(np.array([ self.map_height*np.random.random(size=self.num_particles), self.map_width*np.random.random(size=self.num_particles), 2 * np.pi * np.random.random(size=self.num_particles) ]))
        # Particles are row, col, theta (theta is ROS2 convention)
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

    def update_motion_particles(self, old_transform, new_transform):
        #parallel only, wont work for holonomic robot
        r = old_transform.transform.rotation
        rot = [r.x, r.y, r.z, r.w]
        _, _, odom_angle = euler_from_quaternion(rot)
        parallel_x = np.cos(odom_angle)
        parallel_y = np.sin(odom_angle)
        diff_x = new_transform.transform.translation.x - old_transform.transform.translation.x
        diff_y = new_transform.transform.translation.y - old_transform.transform.translation.y
        odom_diff = np.array([diff_x , diff_y])
        parallel = np.dot(np.array([parallel_x, parallel_y]), odom_diff)
        print("parallel=", parallel, odom_angle, diff_x, diff_y)
        self.particles[:, 1] += parallel * np.cos(self.particles[:,2])/self.resolution
        self.particles[:, 0] -= parallel * np.sin(self.particles[:,2])/self.resolution
        new_r = new_transform.transform.rotation
        new_rot = [new_r.x, new_r.y, new_r.z, new_r.w]
        _, _, new_odom_angle = euler_from_quaternion(new_rot)
        diff_rot = new_odom_angle - odom_angle  # is this valid?
        self.particles[:, 2] += diff_rot

    def predictions(self, map_image, particles):
        image = map_image==0
        trans_coords = np.transpose(self.coord_map + particles[:, :2], axes=(3,2,0,1))
        polar_coord_predictions = ndi.map_coordinates(image, trans_coords, prefilter=False, mode=self.ndi_mode, order=0, cval=0.0)
        skimage.transform._warps._clip_warp_output(image, polar_coord_predictions, 'constant', 0.0, True)
        polar_coords = np.argmax(polar_coord_predictions, axis=2)*self.resolution
        predictions = np.array([ np.flip(np.roll(polar_coords[particle_id], int(360 * self.particles[particle_id, 2] / (2 * np.pi)))) for particle_id in range(len(particles))])
        return predictions

    def resample_particles(self, probs):
        ls = np.array(np.random.choice(np.arange(len(self.particles)), size=self.replacement, p=probs))
        self.particles[:self.replacement, :2] = self.particles[ls][:, :2] + 1 * np.random.normal(size=(self.replacement, 2))
        self.particles[:self.replacement, 2] = self.particles[ls][:, 2] + .1 * np.random.normal(size=(self.replacement))
        new_particles = self.num_particles - self.replacement
        self.particles[self.replacement:] = np.transpose(np.array([ self.map_height*np.random.random(size=new_particles), self.map_width*np.random.random(size=new_particles), 2 * np.pi * np.random.random(size=new_particles) ]))

    def localize(self, scan):
        new_scan = skimage.transform.resize(scan.astype(np.float32), (self.num_angles,))
        new_scan = np.roll(new_scan, -90)  # account for laser mounting.
        predictions = self.predictions(self.map_image, self.particles)
        prediction_error = np.nanmean((predictions - new_scan[np.newaxis, :])**2, axis=1)
        probs = np.exp(-prediction_error)
        probs = probs/probs.sum()
        print("publish...")
        self.resample_particles(probs)

        x_c = np.nanmean(self.particles[:self.replacement, 1])
        y_c = np.nanmean(self.particles[:self.replacement, 0])
        x_std_c = np.sqrt(np.nanmean(self.particles[:self.replacement, 1]**2) - x_c**2)
        y_std_c = np.sqrt(np.nanmean(self.particles[:self.replacement, 0]**2) - y_c**2)
        kappa, angle, _ = vonmises.fit(self.particles[:self.replacement, 2], fscale=1)
        angle_std_c = 1/np.sqrt(kappa)
        mean_predictions = self.predictions(self.map_image, np.array([[y_c, x_c, angle]]))[0]
        x_w = x_c*self.resolution + self.origin[0]
        y_w = (self.map_height-y_c)*self.resolution + self.origin[1]
        x_std_w = x_std_c*self.resolution
        y_std_w = y_std_c*self.resolution

        mean_predictions = self.predictions(self.map_image, np.array([[y_c, x_c, angle]]))[0]

        print("ANGLE=", angle, " A STD=", angle_std_c)
        return (x_w, y_w), angle, x_std_w, y_std_w, angle_std_c, mean_predictions


class Localizer(Node):
    def __init__(self):
        super().__init__("nav")
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
        self.pred_publisher = \
            self.create_publisher(LaserScan, "/pred_laser", 1)
        self.particles_publisher = \
            self.create_publisher(PointCloud2, "/particles", 1)
        self.marker_loc_uncertainty_publisher = self.create_publisher(Marker, 'loc_uncertainty', 1)
        self.angle_uncertainty_publisher = self.create_publisher(Marker, 'angle_uncertainty', 1)
        self.tf_buffer = Buffer()
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True, qos=qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        map = skimage.io.imread(os.path.join(os.path.split(self.map_file)[0], image_filename))
        self.localizer = MCL(map, origin, resolution)
        self.old_transform = None
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

    def send_map_base_link_transform(self, base_link_to_odom_transform, loc, angle, tim):
        try:
            base_laser_to_base_link_transform = self.tf_buffer.lookup_transform(
                "base_laser",
                "base_link",
                rclpy.time.Time())
        except TransformException as ex:
            print("No Transform")
            return
        zero_to_odom_transform = TransformStamped()
        zero_to_odom_transform.header.stamp = self.get_clock().now().to_msg()
        zero_to_odom_transform.header.frame_id = 'zero'
        zero_to_odom_transform.child_frame_id = 'odom'
        zero_to_odom_transform.transform = base_link_to_odom_transform.transform
        self.tf_static_broadcaster.sendTransform(zero_to_odom_transform)

        map_to_zero_transform = TransformStamped()
        map_to_zero_transform.header.stamp = \
            self.get_clock().now().to_msg()
        map_to_zero_transform.header.frame_id = 'map'
        map_to_zero_transform.child_frame_id = 'zero'
        map_to_zero_transform.transform.translation.x = \
            loc[0]
        map_to_zero_transform.transform.translation.y = \
            loc[1]
        map_to_zero_transform.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, angle)
        map_to_zero_transform.transform.rotation.x = q[0]
        map_to_zero_transform.transform.rotation.y = q[1]
        map_to_zero_transform.transform.rotation.z = q[2]
        map_to_zero_transform.transform.rotation.w = q[3]
        self.tf_static_broadcaster.sendTransform(map_to_zero_transform)


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

    def publish_point_cloud(self, header, particles):
        points = np.zeros([self.localizer.replacement, 3])
        points[:, 0] = particles[:self.localizer.replacement, 1]*self.localizer.resolution + self.localizer.origin[0]
        points[:, 1] = (self.localizer.map_height-particles[:self.localizer.replacement, 0])*self.localizer.resolution + self.localizer.origin[1]
        cloud_msg_header = Header()
        cloud_msg_header.stamp = header.stamp
        cloud_msg_header.frame_id = "map"
        cloud_msg = point_cloud2.create_cloud_xyz32(cloud_msg_header, points)
        self.particles_publisher.publish(cloud_msg)

    def publish_loc_uncertainty_marker(self, header, loc, angle, std_x, std_y):
        print("LOC=", std_x)
        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = "base_link"
        marker.ns = "basic_shapes"
        marker.id = 0
        marker.type = 3  # CYLINDER
        marker.action = 0  # ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = std_x
        marker.scale.y = std_y
        marker.scale.z = 0.5
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = .2
        marker.frame_locked = True
        self.marker_loc_uncertainty_publisher.publish(marker)

    def publish_angle_uncertainty_marker(self, header, loc, angle, std_angle):
        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = "base_link"
        marker.ns = "basic"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x = .1
        marker.scale.y = .1
        marker.scale.z = .1
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        mstd_angle = np.min((std_angle, np.pi))
        point1 = Point()
        point1.x, point1.y, point1.z = 0.0, 0.0, 0.1
        point2 = Point()
        point2.x, point2.y, point2.z = 0.0 + .5*np.cos(-mstd_angle), 0.0 + .5*np.sin(-mstd_angle), 0.1
        point3 = Point()
        point3.x, point3.y, point3.z = 0.0 + .5*np.cos(+mstd_angle), 0.0 + .5*np.sin(+mstd_angle), 0.1
        marker.points = [point1, point2, point1, point3]
        marker.frame_locked = True
        self.angle_uncertainty_publisher.publish(marker)

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
        self.localizer.update_motion_particles(self.old_transform, odom_to_base_link_transform)
        loc, angle, std_x, std_y, std_angle, predictions = self.localizer.localize(scan)
        if self.old_transform is None:
            self.old_transform = odom_to_base_link_transform
        self.send_map_base_link_transform(base_link_to_odom_transform, loc, angle, None)
        self.publish_lidar_prediction(lidar_msg.header, predictions)
        self.publish_point_cloud(lidar_msg.header, self.localizer.particles)
        self.publish_loc_uncertainty_marker(lidar_msg.header, loc, angle, std_x, std_y)
        self.publish_angle_uncertainty_marker(lidar_msg.header, loc, angle, std_angle)
        self.old_transform = odom_to_base_link_transform


rclpy.init()
localizer = Localizer()
rclpy.spin(localizer)
nav.destroy_node()
rclpy.shutdown()
