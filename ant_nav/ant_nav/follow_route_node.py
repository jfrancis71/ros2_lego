import rclpy
import scipy.signal
from PIL import Image as PILImage
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from rclpy.time import Time
from cv_bridge import CvBridge
import glob
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2


class AntNav1(Node):
    def __init__(self):
        super().__init__("ant_nav_1")
        self.image_subscription = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            1)
        self.publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.image_publisher = self.create_publisher(Image, "/debug_image", 10)
        self.bridge = CvBridge()
        self.declare_parameter('route_folder', './default_route_folder')
        self.declare_parameter('route_loop', False)
        self.declare_parameter('max_match_threshold', 60.0)
        self.declare_parameter('drive', True)
        self.declare_parameter('lost_seq_len', 5)
        self.declare_parameter('warning_time', .25)
        self.declare_parameter('diagnostic', False)
        self.route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.route_loop = self.get_parameter('route_loop').get_parameter_value().bool_value
        self.max_match_threshold = self.get_parameter('max_match_threshold').get_parameter_value().double_value
        self.drive = self.get_parameter('drive').get_parameter_value().bool_value
        self.lost_seq_len  = self.get_parameter('lost_seq_len').get_parameter_value().integer_value
        self.warning_time = self.get_parameter('warning_time').get_parameter_value().double_value
        self.diagnostic = self.get_parameter('diagnostic').get_parameter_value().bool_value
        self.route_images = self.load_images()
        self.last_image_idx = self.route_images.shape[0]-1
        self.image_idx = 0
        self.lost = self.lost_seq_len
        sld_route_images = np.lib.stride_tricks.sliding_window_view(self.route_images, window_shape=(64, 32, 3), axis=(1, 2, 3))[:, 0, :, 0]
        self.norm_sld_route_images = sld_route_images/sld_route_images.mean(axis=(2,3,4))[:,:,np.newaxis, np.newaxis, np.newaxis]
        if self.diagnostic:
            plt.ion()
            self.fig, self.axs = plt.subplots(1, 5)
        print("Initialized.")

    def normalize(self, image):
        """Binarizes onto (-1,1) using median."""
        return image/image.mean()

    def template_match(self, template, image):
        epsilon = .0000000000001
        template_features = np.lib.stride_tricks.sliding_window_view(template, window_shape=(5, 5), axis=(0, 1))
        template_features = template_features.transpose(0, 1, 3, 4, 2)
        # template feature map is of shape [60, 28, 75] Features are flattened in last dimension as we will be building
        # arrays of feature maps at each position and it may be confusing to have different spatial maps in same array.
        template_features = template_features.reshape(list(template_features.shape[:2]) + [75])
        # We now compute at each point a map of neighbour feature vectors, so we have shape [5, 5, 56, 24, 75]
        template_feature_map = np.lib.stride_tricks.sliding_window_view(template_features, window_shape=(5, 5), axis=(0, 1)).transpose(
            (3, 4, 0, 1, 2))

        image_features = np.lib.stride_tricks.sliding_window_view(image, window_shape=(5, 5), axis=(0, 1))
        image_features = image_features.transpose(0, 1, 3, 4, 2)
        # image_features is of shape [56, 24, 75]
        image_features = image_features.reshape(list(image_features.shape[:2]) + [75])[2:-2, 2:-2]

        red_raw_weight = np.exp(-((image_features[:, :, :36] - template_feature_map[:, :, :, :, :36]) ** 2).sum(axis=-1))
        red_norm_weight = red_raw_weight / (red_raw_weight.sum(axis=(0, 1)) + epsilon)
        red_predictions = (template_feature_map[:, :, :, :, 36] * red_norm_weight).sum(axis=(0, 1))

        green_raw_weight = np.exp(-((image_features[:, :, :37] - template_feature_map[:, :, :, :, :37]) ** 2).sum(axis=-1))
        green_norm_weight = green_raw_weight / (green_raw_weight.sum(axis=(0, 1)) + epsilon)
        green_predictions = (template_feature_map[:, :, :, :, 37] * green_norm_weight).sum(axis=(0, 1))

        blue_raw_weight = np.exp(-((image_features[:, :, :38] - template_feature_map[:, :, :, :, :38]) ** 2).sum(axis=-1))
        blue_norm_weight = blue_raw_weight / (blue_raw_weight.sum(axis=(0, 1)) + epsilon)
        blue_predictions = (template_feature_map[:, :, :, :, 38] * blue_norm_weight).sum(axis=(0, 1))

        return ((red_predictions - image[4:-4, 4:-4, 0]) ** 2).sum() + (
                    (green_predictions - image[4:-4, 4:-4, 1]) ** 2).sum() + (
                    (blue_predictions - image[4:-4, 4:-4, 2]) ** 2).sum()

    def load_images(self):
        """Reads in images resizes to 64x64. Takes subslices of 32x64 and normalizes"""
        files = glob.glob(f"{self.route_folder}/*.jpg")
        files.sort()
        resized = np.array([np.array(PILImage.open(fname).resize((64,64))).astype(np.float32)/256. for fname in files])
        filtered = gaussian_filter(resized, sigma=(0, 1, 1, 0))
        return filtered

    def route_image_diff(self, image):
        centre_image = image[:, 16:48]
        norm_image = self.normalize(centre_image).astype(np.float32)
        diffs = ((norm_image - self.norm_sld_route_images)**2).mean(axis=(2,3,4))
        return diffs

    def publish_twist(self, header, speed, angular_velocity):
        twist_stamped = TwistStamped()
        twist_stamped.header = header
        twist_stamped.twist.linear.x = speed
        twist_stamped.twist.angular.z = angular_velocity
        self.publisher.publish(twist_stamped)

    def von_mises(self, theta0, m, theta):
        c = 1/(2 * np.pi * np.i0(m))
        return c * np.exp(m * np.cos(theta - theta0))

    def lost_q(self, template, image):
        grey_template = np.linalg.norm(template, axis=2)
        grey_image = np.linalg.norm(image, axis=2)
        sobel_x_template = cv2.Sobel(grey_template, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y_template = cv2.Sobel(grey_image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_x_image = cv2.Sobel(grey_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y_image = cv2.Sobel(grey_image, cv2.CV_64F, 0, 1, ksize=5)
        mag_template = np.linalg.norm(np.array([sobel_x_template, sobel_y_template]), axis=0)
        mag_image = np.linalg.norm(np.array([sobel_x_image, sobel_y_image]), axis=0)
        dir_template = np.arctan2(sobel_x_template, sobel_y_template)
        dir_image = np.arctan2(sobel_x_image, sobel_y_image)
        angle_diff_1 = (mag_template > 20.0) * (1 - np.cos(dir_template - dir_image))
        angle_diff_2 = (mag_image > 20.0) * (1 - np.cos(dir_template - dir_image))
        if self.diagnostic:
            axs[0].imshow(template)
            axs[1].imshow(image)
            axs[2].imshow(mag_template, cmap='gray', vmin=0.0, vmax=20.0)
            axs[3].imshow(angle_diff_1, cmap='gray', vmin=0.0, vmax=1.0)
            axs[4].imshow(angle_diff_2, cmap='gray', vmin=0.0, vmax=1.0)
            plt.pause(.001)
        angle_diff_1 = angle_diff_1/(mag_template > 20.0).sum()
        angle_diff_2 = angle_diff_2 / (mag_image > 20.0).sum()
        return (angle_diff_1 + angle_diff_2).sum()

    def get_drive_instructions(self, np_image):
        image = gaussian_filter(np_image, sigma=(1, 1, 0))
        image_diffs = self.route_image_diff(image)
        template_min = np.sqrt(image_diffs.min())
        image_idx, angle = np.unravel_index(np.argmin(image_diffs, axis=None), image_diffs.shape)
        angle = np.argmin(image_diffs[(image_idx + 1) % self.last_image_idx])
        centre_image = image[:, 16:48]
        sub_window_idx = np.argmin(image_diffs[image_idx])
        norm_image = self.normalize(centre_image).astype(np.float32)
        flex_template_min = self.template_match(self.route_images[image_idx, :, sub_window_idx:sub_window_idx+32], norm_image)
        lost = self.lost_q(self.route_images[image_idx, :, sub_window_idx:sub_window_idx+32], norm_image)
        return image_idx, angle-16, template_min, flex_template_min, lost

    def warnings(self, image_msg_timestamp, time_received):
        source_message_time = Time.from_msg(image_msg_timestamp).nanoseconds
        network_transit_time = (time_received - source_message_time)*1e-9
        now = self.get_clock().now().nanoseconds
        process_time = (now - source_message_time)*1e-9
        cpu_time = (now - time_received)*1e-9
        if process_time > self.warning_time:
            warn_msg = f'Delay processing drive instructions of {process_time} with cpu time {cpu_time}, network transit time {network_transit_time}'
            self.get_logger().warn(warn_msg)

    def image_callback(self, image_msg):
        time_received = self.get_clock().now().nanoseconds
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        pil_image = PILImage.fromarray(cv_image)
        image = np.array(pil_image.resize((64,64))).astype(np.float32)/256.
        image_idx, angle, template_min, lost_flex_template_min, lost_edge_min = self.get_drive_instructions(image)
        print(f'matched image idx {image_idx}, angle={angle}, template_min={template_min}, flex={lost_flex_template_min}, edge_min={lost_edge_min}')
        if lost_edge_min > .30:
            self.lost += 1
        else:
            self.lost = 0
        speed = 0.0
        angular_velocity = 0.0
        if self.lost < self.lost_seq_len and (image_idx != self.last_image_idx or self.route_loop):
            speed = 0.05
            angular_velocity = angle/48
            if image_idx == self.last_image_idx-1:
                angular_velocity = 0.0
        if self.drive:
            self.publish_twist(image_msg.header, speed, angular_velocity)
        self.warnings(image_msg.header.stamp, time_received)


rclpy.init()
ant_nav = AntNav1()
rclpy.spin(ant_nav)
ant_nav.destroy_node()
rclpy.shutdown()
