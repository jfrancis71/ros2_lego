import glob
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.stats
from scipy.stats import chi2, vonmises
from PIL import Image as PILImage
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from rclpy.time import Time
from cv_bridge import CvBridge


class LostDetector:
    def __init__(self):
        self.edge_direction_error = None
        self.edge_mag_error = None
        self.heading_text = np.zeros([30, 513, 3]).astype(np.float32)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.heading_text, 'Edge Orientation Error', (60, 20), font, .5, (1, 1, 1), 1, cv2.LINE_AA)
        cv2.putText(self.heading_text, 'Edge Magnitude Error', (280, 20), font, .5, (1, 1, 1), 1, cv2.LINE_AA)

    def lost(self, template, image):
        sobel_x_template = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y_template = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize=5)
        sobel_x_image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y_image = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        mag_template = np.linalg.norm(np.array([sobel_x_template, sobel_y_template]), axis=0)
        mag_image = np.linalg.norm(np.array([sobel_x_image, sobel_y_image]), axis=0)
        dir_template = np.arctan2(sobel_x_template, sobel_y_template)
        dir_image = np.arctan2(sobel_x_image, sobel_y_image)
        m = (1 - np.exp(-mag_template/40))*4
        self.edge_direction_error = vonmises(m, dir_template).logpdf(dir_image) - np.log(1.0/(2*np.pi))
        self.edge_mag_error = (chi2(mag_template+1).logpdf(mag_image+1) - chi2(10).logpdf(mag_image+1))/20
        return self.edge_direction_error.sum() + self.edge_mag_error.sum()

    def diagnostic(self):
        edge_direction_error = self.edge_direction_error.sum(axis=-1)
        edge_mag_error = self.edge_mag_error.sum(axis=-1)
        resized_angle_error = cv2.resize(edge_direction_error, (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_mag_error = cv2.resize(edge_mag_error, (256, 256), interpolation=cv2.INTER_NEAREST)
        diagnostic_angle_error = np.zeros([256, 256, 3]).astype(np.float32)
        diagnostic_angle_error[:, :, 2] = np.clip(resized_angle_error/2.0, 0.0, 1.0)
        diagnostic_angle_error[:, :, 0] = np.clip(-resized_angle_error / 2.0, 0.0, 1.0)
        diagnostic_mag_error = np.zeros([256, 256, 3]).astype(np.float32)
        diagnostic_mag_error[:, :, 0] = np.clip(-resized_mag_error, 0.0, 1.0)
        diagnostic_mag_error[:, :, 2] = np.clip(resized_mag_error, 0.0, 1.0)
        cv2.line(diagnostic_angle_error, (8*16, 0), (8*16, 256), color=(0, .7, .7))
        cv2.line(diagnostic_mag_error, (8 * 16, 0), (8 * 16, 256), color=(0, .7, .7))
        divider = np.zeros([256, 1, 3]).astype(np.float32)
        divider[:, :, 1] = 1.0
        diagnostic_image = cv2.hconcat([diagnostic_angle_error, divider, diagnostic_mag_error])
        cv2.line(diagnostic_image, (0, 16 * 8), (512, 16 * 8), color=(0, .7, .7))
        canvas = cv2.vconcat([self.heading_text, diagnostic_image])
        return canvas


class CatNav(Node):
    def __init__(self):
        super().__init__("cat_nav")
        self.declare_parameter('route_folder', './default_route_folder')
        self.declare_parameter('route_loop', False)
        self.declare_parameter('lost_edge_threshold', 450.0)
        self.declare_parameter('drive', True)
        self.declare_parameter('lost_seq_len', 5)
        self.declare_parameter('warning_time', .25)
        self.declare_parameter("publish_diagnostic", True)
        self.declare_parameter('angle_ratio', 36.)
        self.declare_parameter('stop_on_last', 5)
        self.declare_parameter('forward_speed', .05)
        self.route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.route_loop = self.get_parameter('route_loop').get_parameter_value().bool_value
        self.lost_edge_threshold = self.get_parameter('lost_edge_threshold').get_parameter_value().double_value
        self.drive = self.get_parameter('drive').get_parameter_value().bool_value
        self.lost_seq_len  = self.get_parameter('lost_seq_len').get_parameter_value().integer_value
        self.warning_time = self.get_parameter('warning_time').get_parameter_value().double_value
        self.angle_ratio = self.get_parameter('angle_ratio').get_parameter_value().double_value
        self.stop_on_last = self.get_parameter('stop_on_last').get_parameter_value().integer_value
        self.forward_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        self.blur = 1
        self.route_images = self.load_images()
        filtered = gaussian_filter(self.route_images, sigma=(0, self.blur, self.blur, 0))
        sld_route_images = np.lib.stride_tricks.sliding_window_view(filtered, window_shape=(32, 16, 3), axis=(1, 2, 3))[:, 0, :, 0]
        self.norm_sld_route_images = sld_route_images/sld_route_images.mean(axis=(2,3,4))[:,:,np.newaxis, np.newaxis, np.newaxis]
        self.last_image_idx = self.route_images.shape[0]-1
        self.image_idx = 0
        self.lost = self.lost_seq_len
        self.image_subscription = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            1)
        self.twist_publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        if self.get_parameter('publish_diagnostic').get_parameter_value().bool_value:
            self.diagnostic_image_publisher = self.create_publisher(Image, "/diagnostic_image", 10)
            self.heading_text = np.zeros([30, 513, 3]).astype(np.float32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.heading_text, 'Current Camera Image', (60, 20), font, .5, (1, 1, 1), 1, cv2.LINE_AA)
            cv2.putText(self.heading_text, 'Best Template Match', (280, 20), font, .5, (1, 1, 1), 1, cv2.LINE_AA)
        else:
            self.diagnostic_image_publisher = None
        self.bridge = CvBridge()
        self.lost_detector = LostDetector()
        print("Initialized.")

    def load_images(self):
        files = glob.glob(f"{self.route_folder}/*.jpg")
        files.sort()
        resized = np.array([np.array(PILImage.open(fname).resize((32,32))).astype(np.float32)/256. for fname in files])
        return resized

    def publish_twist(self, header, speed, angular_velocity):
        twist_stamped = TwistStamped()
        twist_stamped.header = header
        twist_stamped.twist.linear.x = speed
        twist_stamped.twist.angular.z = angular_velocity
        self.twist_publisher.publish(twist_stamped)

    def diagnostic_image(self, image, template):
        resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_template = cv2.resize(template, (256, 256), interpolation=cv2.INTER_NEAREST)
        divider = np.zeros([256, 1, 3]).astype(np.float32)
        divider[:, :, 1] = 1.0
        cv2.line(resized_image, (8*16, 0), (8*16, 256), color=(0, .7, .7))
        cv2.line(resized_template, (8 * 16, 0), (8 * 16, 256), color=(0, .7, .7))
        top = cv2.hconcat([resized_image, divider, resized_template])
        cv2.line(top, (0, 16*8), (512, 16*8), color=(0, .7, .7))
        lost_diagnostic_image = self.lost_detector.diagnostic()
        canvas = cv2.vconcat([self.heading_text, top, lost_diagnostic_image])
        return canvas

    def get_drive_instructions(self, image):
        image_diffs = ((image - self.norm_sld_route_images)**2).mean(axis=(2,3,4))
        template_min = np.sqrt(image_diffs.min())
        image_idx, angle = np.unravel_index(np.argmin(image_diffs, axis=None), image_diffs.shape)
        angle = np.argmin(image_diffs[(image_idx + 1) % self.last_image_idx])
        sub_window_idx = np.argmin(image_diffs[image_idx])
        return image_idx, sub_window_idx, angle-8, template_min

    def timing_warnings(self, image_msg_timestamp, time_received):
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
        image = np.array(pil_image.resize((32,32))).astype(np.float32)/256.
        smoothed_image = gaussian_filter(image, sigma=(self.blur, self.blur, 0))
        centre_image = smoothed_image[:, 8:24].astype(np.float32)
        norm_image = centre_image/centre_image.mean()
        image_idx, sub_window_idx, angle, template_min = self.get_drive_instructions(norm_image)
        lost_edge_min = self.lost_detector.lost(self.norm_sld_route_images[image_idx, sub_window_idx], norm_image)
        print(f'matched image idx {image_idx}, angle={angle}, template_min={template_min:.2f}, lost_edge={lost_edge_min:.2f}')
        if lost_edge_min < self.lost_edge_threshold:
            self.lost += 1
        else:
            self.lost = 0
        if self.drive and self.lost < self.lost_seq_len and (image_idx < self.last_image_idx-self.stop_on_last or self.route_loop):
            speed = self.forward_speed
            angular_velocity = angle/self.angle_ratio
            self.publish_twist(image_msg.header, speed, angular_velocity)
        if self.diagnostic_image_publisher:
            diagnostic_image = self.diagnostic_image(image[:, 8:24], self.route_images[image_idx, :,  sub_window_idx:sub_window_idx+16])
            diagnostic_image_msg = self.bridge.cv2_to_imgmsg((diagnostic_image*255).astype(np.uint8),
                                                       encoding="rgb8")
            self.diagnostic_image_publisher.publish(diagnostic_image_msg)
        self.timing_warnings(image_msg.header.stamp, time_received)


rclpy.init()
cat_nav = CatNav()
rclpy.spin(cat_nav)
cat_nav.destroy_node()
rclpy.shutdown()
