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
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from scipy import ndimage as ndi
from skimage._shared.utils import _to_ndimage_mode
from skimage._shared.utils import convert_to_float


class DiscreteLocalizer:
    def __init__(self, map_image, origin, resolution, num_degrees, num_x_resolution, num_y_resolution):
        self.num_degrees = num_degrees
        self.resolution = resolution
        self.origin = origin
        self.centers = [[(r,c) for c in range(0, map_image.shape[1], int(num_x_resolution/resolution))] for r in range(0, map_image.shape[0], int(num_y_resolution/resolution))]
        self.centers = list(itertools.chain(*self.centers))
        self.trans = np.array([skimage.transform.warp_polar(map_image==0, center=c, radius=100, output_shape=(num_degrees, 100)) for c in self.centers])
        self.polar_coords = np.array([np.argmax(self.trans[c], axis=1)*resolution for c in range(len(self.centers))])
        self.map_height = map_image.shape[0]

    def localize(self, scan):
        new_scan = skimage.transform.resize(scan.astype(np.float32), (self.num_degrees,))
        predictions = np.array([[np.roll(np.flip(self.polar_coords[c]), s) for s in range(self.num_degrees)] for c in range(len(self.centers))])

        angles = np.nanmean((predictions - new_scan)**2, axis=0)[np.newaxis, np.newaxis]
        prediction_error = np.nanmean((predictions - new_scan)**2, axis=2)
        idx = np.unravel_index(np.argmin(prediction_error), prediction_error.shape)
        loc = idx[0]
        angle = 2 * np.pi * idx[1]/self.num_degrees
        return (self.centers[loc][1]*self.resolution + self.origin[0], (self.map_height-self.centers[loc][0])*self.resolution + self.origin[1]), angle, predictions[loc][idx[1]]


class MCL:
    def __init__(self, map_image, origin, resolution):
        self.map_image = map_image
        self.origin = origin
        self.resolution = resolution
        self.map_height = map_image.shape[0]
        self.map_width = map_image.shape[1]
        self.particles = [(self.map_height*np.random.random(), self.map_width*np.random.random()) for m in range(150)]
        self.particles = np.transpose(np.array([ self.map_height*np.random.random(size=150), self.map_width*np.random.random(size=150) ]))
        height = 360
        k_radius = 100 / 100
        k_angle = height / (2 * np.pi)
        def coord_map1(output_coords):
            angle = output_coords[:, 1] / k_angle
            rr = ((output_coords[:, 0] / k_radius) * np.sin(angle))
            cc = ((output_coords[:, 0] / k_radius) * np.cos(angle))
            coords = np.column_stack((cc, rr))
            return coords
        self.c = skimage.transform.warp_coords(coord_map1, (360, 100))
        self.wa = np.transpose(self.c, axes=(1, 2, 0))[:, :, np.newaxis, :]

    def predictions(self, map_image, particles):
        ndi_mode = _to_ndimage_mode('constant')
        image = map_image==0
        coords1 = np.transpose(self.wa + self.particles, axes=(3,2,0,1))
        warpd = ndi.map_coordinates(image, coords1, prefilter=False, mode=ndi_mode, order=0, cval=0.0)
        skimage.transform._warps._clip_warp_output(image, warpd, 'constant', 0.0, True)
        return warpd

    def localize(self, scan):
        new_scan = skimage.transform.resize(scan.astype(np.float32), (360,))
        trans = self.predictions(self.map_image, self.particles)
        polar_coords = np.argmax(trans, axis=2)*self.resolution
        predictions = np.transpose(circulant(np.flip(polar_coords, axis=1)), axes=(0, 2, 1))
        prediction_error = np.nanmean((predictions - new_scan)**2, axis=2)
        probs = np.exp(-prediction_error)
        idx = np.unravel_index(np.argmin(prediction_error), prediction_error.shape)
        loc = idx[0]
        angle = 2 * np.pi * idx[1]/360
        x = self.particles[loc][1]*self.resolution + self.origin[0]
        y = (self.map_height-self.particles[loc][0])*self.resolution + self.origin[1]
        norm = probs.max(axis=1)
        norm_1 = norm/norm.sum()
        print("publish...")
        ls = np.array(np.random.choice(np.arange(len(self.particles)), size=150, p=norm_1))
        self.particles = self.particles[ls] + np.random.normal(size=(150, 2))
        return (x, y), angle, predictions[loc][idx[1]]


class Localizer(Node):
    def __init__(self):
        super().__init__("nav")
        self.declare_parameter('map', 'my_house.yaml')
        self.declare_parameter('discrete_num_degrees', 360)
        self.declare_parameter('discrete_num_x_resolution', .05)
        self.declare_parameter('discrete_num_y_resolution', .05)
        self.map_file = self.get_parameter('map').get_parameter_value().string_value
        self.discrete_num_degrees = self.get_parameter('discrete_num_degrees').get_parameter_value().integer_value
        discrete_num_x_resolution = self.get_parameter('discrete_num_x_resolution').get_parameter_value().double_value
        discrete_num_y_resolution = self.get_parameter('discrete_num_x_resolution').get_parameter_value().double_value

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
            self.create_publisher(LaserScan, "/pred_laser", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        map = skimage.io.imread(os.path.join(os.path.split(self.map_file)[0], image_filename))
#        self.localizer = DiscreteLocalizer(map, origin, resolution, self.discrete_num_degrees, discrete_num_x_resolution, discrete_num_y_resolution)
#        self.localizer = MetropolisLocalizer(map, origin, resolution)
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
        lidar_msg1.angle_increment = 2 * np.pi / self.discrete_num_degrees
        lidar_msg1.time_increment = 0.00019850002718158066
        lidar_msg1.scan_time = 0.10004401206970215
        lidar_msg1.range_min = 0.019999999552965164
        lidar_msg1.range_max = 25.0
        lidar_msg1.header = header
        self.pred_publisher.publish(lidar_msg1)

    def lidar_callback(self, lidar_msg):
        scan = np.array(lidar_msg.ranges)
        loc, angle, predictions = self.localizer.localize(scan)
        self.send_map_base_link_transform(loc, angle, None)
        self.publish_lidar_prediction(lidar_msg.header, predictions)


rclpy.init()
localizer = Localizer()
rclpy.spin(localizer)
nav.destroy_node()
rclpy.shutdown()
