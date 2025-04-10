import rclpy
from PIL import Image as PILImage
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
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
            10)
        self.publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.image_publisher = self.create_publisher(Image, "/debug_image", 10)
        self.bridge = CvBridge()
        self.declare_parameter('route_folder', './default_route_folder')
        self.no_logging = "NoLogging"
        self.declare_parameter('log_folder', self.no_logging)
        self.declare_parameter('route_loop', False)
        self.declare_parameter('max_match_threshold', 80.0)
        self.declare_parameter('drive', True)
        self.route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.log_folder = self.get_parameter('log_folder').get_parameter_value().string_value
        self.route_loop = self.get_parameter('route_loop').get_parameter_value().bool_value
        self.max_match_threshold = self.get_parameter('max_match_threshold').get_parameter_value().double_value
        self.drive = self.get_parameter('drive').get_parameter_value().bool_value
        self.images = self.load_images()
        self.last_image_idx = self.images.shape[0]-1
        self.image_idx = 0

    def normalize(self, image):
        """Binarizes onto (-1,1) using median."""
        return image/image.mean()

    def template_match(self, template, image):
        template_sliding = np.lib.stride_tricks.sliding_window_view(template, window_shape=(5, 5), axis=(0, 1))
        template_shape = template_sliding.shape
        t = template_sliding.reshape(list(template_shape[:2]) + [75])
        template = np.lib.stride_tricks.sliding_window_view(t, window_shape=(5, 5), axis=(0, 1)).transpose(
            (3, 4, 0, 1, 2))
        # return template

        obj_sliding = np.lib.stride_tricks.sliding_window_view(image, window_shape=(5, 5), axis=(0, 1))
        obj_shape = obj_sliding.shape
        obj = obj_sliding.reshape(list(obj_shape[:2]) + [75])[2:-2, 2:-2]

        red_raw_weight = np.exp(-((obj[:, :, :36] - template[:, :, :, :, :36]) ** 2).sum(axis=-1))
        red_norm_weight = red_raw_weight / (red_raw_weight.sum(axis=(0, 1)) + .0000000000001)
        red_predictions = (template_sliding[2:-2, 2:-2, 0].transpose((2, 3, 0, 1)) * red_norm_weight).sum(axis=(0, 1))

        green_raw_weight = np.exp(-((obj[:, :, :37] - template[:, :, :, :, :37]) ** 2).sum(axis=-1))
        green_norm_weight = green_raw_weight / (green_raw_weight.sum(axis=(0, 1)) + .0000000000001)
        green_predictions = (template_sliding[2:-2, 2:-2, 1].transpose((2, 3, 0, 1)) * green_norm_weight).sum(
            axis=(0, 1))

        blue_raw_weight = np.exp(-((obj[:, :, :38] - template[:, :, :, :, :38]) ** 2).sum(axis=-1))
        blue_norm_weight = blue_raw_weight / (blue_raw_weight.sum(axis=(0, 1)) + .0000000000001)
        blue_predictions = (template_sliding[2:-2, 2:-2, 2].transpose((2, 3, 0, 1)) * blue_norm_weight).sum(axis=(0, 1))

        return ((red_predictions - image[4:-4, 4:-4, 0]) ** 2).sum() + (
                    (green_predictions - image[4:-4, 4:-4, 1]) ** 2).sum() + (
                    (blue_predictions - image[4:-4, 4:-4, 2]) ** 2).sum()

    def load_images(self):
        """Reads in images resizes to 64x64. Takes subslices of 32x64 and normalizes"""
        files = glob.glob(f"{self.route_folder}/*.jpg")
        files.sort()
        self.resized = np.array([np.array(PILImage.open(fname).resize((64,64)))/256. for fname in files])
        normalized = np.array([np.array([self.normalize(self.resized[image_idx, :, offset:32+offset]) for offset in range(32)]) for image_idx in range(len(files))])
        normalized = gaussian_filter(normalized, sigma=(0, 0, 2, 2, 0))
        return normalized.astype(np.float32)

    def save_image(self, image):
        if self.log_folder is not self.no_logging:
            self.image_idx += 1
            image.save(f"{self.log_folder}/{self.image_idx:04d}.jpg")
            print("Saving image", self.image_idx)

    def route_image_diff(self, image):
        centre_image = image[:, 16:48]
        norm_image = self.normalize(centre_image).astype(np.float32)
        start = time.time()
        diffs = ((norm_image - self.images)**2).mean(axis=(2,3,4))
        end = time.time()
        duration = end - start
        if (duration > .1):
            warn_msg = f'Delay computing route_image_diff {duration}'
            self.get_logger().warn(warn_msg)
        return diffs

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        pil_image = PILImage.fromarray(cv_image)
        image = np.array(pil_image.resize((64,64))).astype(np.float32)/256.
        image = gaussian_filter(image, sigma=(2, 2, 0))
        start = time.time()
        image_diffs = self.route_image_diff(image)
        end = time.time()
        duration = end - start
        if (duration > .1):
            warn_msg = f'Delay computing diff of {duration}'
            self.get_logger().warn(warn_msg)
        twist_stamped = TwistStamped()
        twist_stamped.header = image_msg.header
        debug_image_msg = self.bridge.cv2_to_imgmsg((image_diffs.clip(0.0, 1.0)*256).astype(np.int8), "8SC1")
        cmin = np.sqrt(image_diffs.min())
        image_idx, angle = np.unravel_index(np.argmin(image_diffs, axis=None), image_diffs.shape)
        centre_image = image[:, 16:48]
        norm_image = self.normalize(centre_image).astype(np.float32)
        sub_window_idx = np.argmin(image_diffs[image_idx])
        flex_diff = self.template_match(self.images[image_idx, sub_window_idx], norm_image)
        angle = np.argmin(image_diffs[(image_idx+1) % self.last_image_idx])
        angle = angle-16
        print("image_idx:", image_idx, ", angle: ", angle, "cmin=", cmin, "flex diff=", flex_diff)
        if flex_diff > self.max_match_threshold or (self.route_loop is False and image_idx == self.last_image_idx):
            twist_stamped.twist.linear.x = 0.00
            twist_stamped.twist.angular.z = 0.00
        else:
            twist_stamped.twist.linear.x = 0.05
            twist_stamped.twist.angular.z = angle/48
        if self.drive:
            self.publisher.publish(twist_stamped)
        self.image_publisher.publish(debug_image_msg)
        self.save_image(pil_image)


rclpy.init()
ant_nav = AntNav1()
rclpy.spin(ant_nav)
ant_nav.destroy_node()
rclpy.shutdown()
