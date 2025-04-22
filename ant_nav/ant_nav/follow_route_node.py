import rclpy
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
        self.declare_parameter('lost_folder', './default_lost_route_folder')
        self.no_logging = "NoLogging"
        self.declare_parameter('log_folder', self.no_logging)
        self.declare_parameter('route_loop', False)
        self.declare_parameter('max_match_threshold', 60.0)
        self.declare_parameter('drive', True)
        self.declare_parameter('lost_seq_len', 5)
        self.declare_parameter('warning_time', .25)
        self.route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.lost_folder = self.get_parameter('lost_folder').get_parameter_value().string_value
        self.log_folder = self.get_parameter('log_folder').get_parameter_value().string_value
        self.route_loop = self.get_parameter('route_loop').get_parameter_value().bool_value
        self.max_match_threshold = self.get_parameter('max_match_threshold').get_parameter_value().double_value
        self.drive = self.get_parameter('drive').get_parameter_value().bool_value
        self.lost_seq_len  = self.get_parameter('lost_seq_len').get_parameter_value().integer_value
        self.warning_time = self.get_parameter('warning_time').get_parameter_value().double_value
        self.images = self.load_images()
        self.lost_images = self.load_lost_images()
        self.last_image_idx = self.images.shape[0]-1
        self.image_idx = 0
        self.lost = 0

        center_images = self.images[:, 15]
        sliding = np.lib.stride_tricks.sliding_window_view(center_images, window_shape=(5, 5), axis=(1, 2)).transpose(0, 1, 2, 4, 5, 3)
        self.reshape = sliding.reshape([center_images.shape[0] * 60 * 28, 3 * 5 * 5])
        # feature map is of shape [#images, 60, 28, 75]
        self.feature_map = np.random.permutation(self.reshape)

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

    def template_lost(self, image):
        epsilon = .0000000000001
        image_features = np.lib.stride_tricks.sliding_window_view(image, window_shape=(5, 5), axis=(0, 1))
        image_features = image_features.transpose(0, 1, 3, 4, 2)
        # image_features is of shape [56, 24, 75]
        image_features = image_features.reshape(list(image_features.shape[:2]) + [75])[2:-2, 2:-2]

        limit_ims = 100

        red_raw_weight = np.exp(-((image_features[:, :, np.newaxis, :36] - self.feature_map[:limit_ims, :36]) ** 2).sum(axis=-1))
        red_norm_weight = red_raw_weight / (red_raw_weight.sum(axis=2) + epsilon)[:, :, np.newaxis]
        red_predictions = (self.feature_map[:limit_ims, 36] * red_norm_weight).sum(axis=2)

        green_raw_weight = np.exp(-((image_features[:, :, np.newaxis, :37] - self.feature_map[:limit_ims, :37]) ** 2).sum(axis=-1))
        green_norm_weight = green_raw_weight / (green_raw_weight.sum(axis=2) + epsilon)[:, :, np.newaxis]
        green_predictions = (self.feature_map[:limit_ims, 37] * green_norm_weight).sum(axis=2)

        blue_raw_weight = np.exp(-((image_features[:, :, np.newaxis, :38] - self.feature_map[:limit_ims, :38]) ** 2).sum(axis=-1))
        blue_norm_weight = blue_raw_weight / (blue_raw_weight.sum(axis=2) + epsilon)[:, :, np.newaxis]
        blue_predictions = (self.feature_map[:limit_ims, 38] * blue_norm_weight).sum(axis=2)

        return ((red_predictions - image[4:-4, 4:-4, 0]) ** 2).sum() + (
                    (green_predictions - image[4:-4, 4:-4, 1]) ** 2).sum() + (
                    (blue_predictions - image[4:-4, 4:-4, 2]) ** 2).sum()


    def load_images(self):
        """Reads in images resizes to 64x64. Takes subslices of 32x64 and normalizes"""
        files = glob.glob(f"{self.route_folder}/*.jpg")
        files.sort()
        resized = np.array([np.array(PILImage.open(fname).resize((64,64)))/256. for fname in files])
        normalized = np.array([np.array([self.normalize(resized[image_idx, :, offset:32+offset]) for offset in range(32)]) for image_idx in range(len(files))])
        normalized = gaussian_filter(normalized, sigma=(0, 0, 2, 2, 0))
        return normalized.astype(np.float32)

    def load_lost_images(self):
        """Reads in images resizes to 64x64. Takes subslices of 32x64 and normalizes"""
        files = glob.glob(f"{self.lost_folder}/*.jpg")
        files.sort()
        resized = np.array([np.array(PILImage.open(fname).resize((64,64)))/256. for fname in files])
        normalized = np.array([np.array([self.normalize(resized[image_idx, :, offset:32+offset]) for offset in range(32)]) for image_idx in range(len(files))])[:,15]
        normalized = gaussian_filter(normalized, sigma=(0, 2, 2, 0))
        return normalized.astype(np.float32)

    def save_image(self, image):
        self.image_idx += 1
        image.save(f"{self.log_folder}/{self.image_idx:04d}.jpg")
        print("Saving image", self.image_idx)

    def route_image_diff(self, image):
        centre_image = image[:, 16:48]
        norm_image = self.normalize(centre_image).astype(np.float32)
        diffs = ((norm_image - self.images)**2).mean(axis=(2,3,4))
        return diffs

    def lost_route_image_diff(self, image):
        centre_image = image[:, 16:48]
        norm_image = self.normalize(centre_image).astype(np.float32)
        diffs = ((norm_image - self.lost_images)**2).mean(axis=(1,2,3))
        return diffs

    def publish_twist(self, header, speed, angular_velocity):
        twist_stamped = TwistStamped()
        twist_stamped.header = header
        twist_stamped.twist.linear.x = speed
        twist_stamped.twist.angular.z = angular_velocity
        self.publisher.publish(twist_stamped)

    def get_drive_instructions(self, np_image):
        image = gaussian_filter(np_image, sigma=(2, 2, 0))
        image_diffs = self.route_image_diff(image)
        lost_image_diffs = self.lost_route_image_diff(image)
        template_min = np.sqrt(image_diffs.min())
        image_idx, angle = np.unravel_index(np.argmin(image_diffs, axis=None), image_diffs.shape)
        angle = np.argmin(image_diffs[(image_idx + 1) % self.last_image_idx])
        centre_image = image[:, 16:48]
        sub_window_idx = np.argmin(image_diffs[image_idx])
        norm_image = self.normalize(centre_image).astype(np.float32)
        flex_template_min = self.template_match(self.images[image_idx, sub_window_idx], norm_image)
        lost_min = np.sqrt(lost_image_diffs.min())
        lost_image_idx = np.unravel_index(np.argmin(lost_image_diffs, axis=None), lost_image_diffs.shape)
        lost_flex_template_min = self.template_lost(norm_image)
        return image_idx, angle-16, template_min, lost_min, flex_template_min, lost_flex_template_min, lost_image_idx

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
        image_idx, angle, template_min, lost_template_min, flex_template_min, lost_flex_template_min, lost_image_idx = self.get_drive_instructions(image)
        print("image_idx:", image_idx, ", angle: ", angle, "template_min=", template_min, "lost_template_min min=",
              lost_template_min, " flex template min", flex_template_min, " lost template flex min", lost_flex_template_min, " lost image idx = ", lost_image_idx)
        if lost_template_min < template_min:
            self.lost += 1
        else:
            self.lost = 0
        speed = 0.0
        angular_velocity = 0.0
        if self.lost < self.lost_seq_len and (image_idx != self.last_image_idx or self.route_loop):
            speed = 0.05
            angular_velocity = angle/48
        if self.drive:
            self.publish_twist(image_msg.header, speed, angular_velocity)
        self.warnings(image_msg.header.stamp, time_received)
        if self.log_folder is not self.no_logging:
            self.save_image(pil_image)


rclpy.init()
ant_nav = AntNav1()
rclpy.spin(ant_nav)
ant_nav.destroy_node()
rclpy.shutdown()
