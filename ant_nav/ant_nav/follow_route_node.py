import rclpy
from PIL import Image as PILImage
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import glob
import time


class AntNav1(Node):
    def __init__(self):
        super().__init__("ant_nav_1")
        self.image_subscription = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            10)
        self.publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.bridge = CvBridge()
        self.declare_parameter('route_folder', './default_route_folder')
        self.no_logging = "NoLogging"
        self.declare_parameter('log_folder', self.no_logging)
        self.declare_parameter('route_loop', False)
        self.route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.log_folder = self.get_parameter('log_folder').get_parameter_value().string_value
        self.route_loop = self.get_parameter('route_loop').get_parameter_value().bool_value
        self.images = self.load_images()
        self.last_image_idx = self.images.shape[0]-1
        self.image_idx = 0


    def load_images(self):
        """Reads in images resizes to 64x64 and normalizes"""
        files = glob.glob(f"{self.route_folder}/*.jpg")
        files.sort()
        resized = np.array([np.array(PILImage.open(file_name).resize((64,64))).astype(np.float32)/256. for file_name in files])
        normalized = (resized.transpose(1, 2, 3, 0)/resized.mean(axis=(1,2,3))).transpose(3, 0, 1, 2)
        return normalized

    def save_image(self, image):
        self.image_idx += 1
        image.save(f"{self.log_folder}/{self.image_idx:04d}.jpg")
        print("Saving image", self.image_idx)

    def template_match(self, templates, image):
        return ((image - templates) ** 2).mean(axis=(1, 2, 3))

    def route_image_diff(self, image):
        norm_image = image/image.mean()
        start = time.time()
        diffs = self.template_match(self.images, norm_image)
        end = time.time()
        duration = end - start
        if duration > .1:
            warn_msg = f'Delay computing route_image_diff {duration}'
            self.get_logger().warn(warn_msg)
        return diffs

    def calc_offset(self, template, image):
        image_centre = image[:, 16:-16]
        offsets = self.template_match(np.array([ template[:, offset:offset+32] for offset in range(32)]), image_centre)
        return offsets

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        pil_image = PILImage.fromarray(cv_image)
        resized = np.array(pil_image.resize((64,64))).astype(np.float32)/256.
        image = resized/resized.mean()
        start = time.time()
        route_diffs = self.route_image_diff(image)
        cmin = route_diffs.min()
        image_idx = route_diffs.argmin()
        offsets = self.calc_offset(self.images[(image_idx + 1) % self.last_image_idx], image)
        angle = offsets.argmin()
        angle = angle - 16
        end = time.time()
        duration = end - start
        if duration > .1:
            warn_msg = f'Delay computing diff of {duration}'
            self.get_logger().warn(warn_msg)
        twist_stamped = TwistStamped()
        twist_stamped.header = image_msg.header

        print("image_idx:", image_idx, ", angle: ", angle, "cmin=", cmin)
        if cmin > 0.2 or (self.route_loop is False and image_idx == self.last_image_idx):
            twist_stamped.twist.linear.x = 0.00
            twist_stamped.twist.angular.z = 0.00
        else:
            twist_stamped.twist.linear.x = 0.05
            twist_stamped.twist.angular.z = angle/48
        self.publisher.publish(twist_stamped)
        if self.log_folder is not self.no_logging:
            self.save_image(pil_image)


rclpy.init()
ant_nav = AntNav1()
rclpy.spin(ant_nav)
ant_nav.destroy_node()
rclpy.shutdown()
