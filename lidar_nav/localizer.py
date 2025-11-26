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
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from scipy import ndimage as ndi
from skimage._shared.utils import _to_ndimage_mode
from skimage._shared.utils import convert_to_float


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

    def predictions(self, map_image, particles):
        image = map_image==0
        trans_coords = np.transpose(self.coord_map + particles[:, :2], axes=(3,2,0,1))
        polar_coord_predictions = ndi.map_coordinates(image, trans_coords, prefilter=False, mode=self.ndi_mode, order=0, cval=0.0)
        skimage.transform._warps._clip_warp_output(image, polar_coord_predictions, 'constant', 0.0, True)
        polar_coords = np.argmax(polar_coord_predictions, axis=2)*self.resolution
        predictions = np.array([ np.flip(np.roll(polar_coords[particle_id], -int(360 * self.particles[particle_id, 2] / (2 * np.pi)))) for particle_id in range(self.num_particles)])
        return predictions

    def localize(self, scan):
        new_scan = skimage.transform.resize(scan.astype(np.float32), (self.num_angles,))
        predictions = self.predictions(self.map_image, self.particles)
        prediction_error = np.nanmean((predictions - new_scan[np.newaxis, :])**2, axis=1)
        probs = np.exp(-prediction_error)
        idx = np.argmin(prediction_error)
        angle = self.particles[idx][2]
        x = self.particles[idx][1]*self.resolution + self.origin[0]
        y = (self.map_height-self.particles[idx][0])*self.resolution + self.origin[1]
        norm = probs
        norm_1 = norm/norm.sum()
        print("publish...")
        ls = np.array(np.random.choice(np.arange(len(self.particles)), size=self.replacement, p=norm_1))
        self.particles[:self.replacement, :2] = self.particles[ls][:, :2] + 1 * np.random.normal(size=(self.replacement, 2))
        self.particles[:self.replacement, 2] = self.particles[ls][:, 2] + .1 * np.random.normal(size=(self.replacement))
        new_particles = self.num_particles - self.replacement
        self.particles[self.replacement:] = np.transpose(np.array([ self.map_height*np.random.random(size=new_particles), self.map_width*np.random.random(size=new_particles), 2 * np.pi * np.random.random(size=new_particles) ]))
        return (x, y), angle, predictions[idx]


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
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        map = skimage.io.imread(os.path.join(os.path.split(self.map_file)[0], image_filename))
        self.localizer = MCL(map, origin, resolution)

    def send_map_base_link_transform(self, loc, angle, tim):
        try:
            base_link_to_odom_transform = self.tf_buffer.lookup_transform(
                "base_link",
                "odom",
                rclpy.time.Time())
        except TransformException as ex:
            print("No Transform")
            return
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
        self.tf_broadcaster.sendTransform(zero_to_odom_transform)

        map_base_laser_to_zero_transform = TransformStamped()
        map_base_laser_to_zero_transform.header.stamp = \
            self.get_clock().now().to_msg()
        map_base_laser_to_zero_transform.header.frame_id = 'map_base_laser'
        map_base_laser_to_zero_transform.child_frame_id = 'zero'
        map_base_laser_to_zero_transform.transform = \
            base_laser_to_base_link_transform.transform
        self.tf_broadcaster.sendTransform(map_base_laser_to_zero_transform)

        map_to_map_base_laser_transform = TransformStamped()
        map_to_map_base_laser_transform.header.stamp = \
            self.get_clock().now().to_msg()
        map_to_map_base_laser_transform.header.frame_id = 'map'
        map_to_map_base_laser_transform.child_frame_id = 'map_base_laser'
        map_to_map_base_laser_transform.transform.translation.x = \
            loc[0]
        map_to_map_base_laser_transform.transform.translation.y = \
            loc[1]
        map_to_map_base_laser_transform.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, -angle)
        map_to_map_base_laser_transform.transform.rotation.x = q[0]
        map_to_map_base_laser_transform.transform.rotation.y = q[1]
        map_to_map_base_laser_transform.transform.rotation.z = q[2]
        map_to_map_base_laser_transform.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(map_to_map_base_laser_transform)


    def publish_lidar_prediction(self, header, ranges):
        lidar_msg1 = LaserScan()
        lidar_msg1.ranges = ranges
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
        points = np.zeros([particles.shape[0], 3])
        points[:, 0] = particles[:, 1]*self.localizer.resolution + self.localizer.origin[0]
        points[:, 1] = (self.localizer.map_height-particles[:, 0])*self.localizer.resolution + self.localizer.origin[1]
        cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
        cloud_msg.header.frame_id = "map"
        self.particles_publisher.publish(cloud_msg)

    def lidar_callback(self, lidar_msg):
        scan = np.array(lidar_msg.ranges)
        loc, angle, predictions = self.localizer.localize(scan)
        self.send_map_base_link_transform(loc, angle, None)
        self.publish_lidar_prediction(lidar_msg.header, predictions)
        self.publish_point_cloud(lidar_msg.header, self.localizer.particles)


rclpy.init()
localizer = Localizer()
rclpy.spin(localizer)
nav.destroy_node()
rclpy.shutdown()
