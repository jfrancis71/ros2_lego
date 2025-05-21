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


class SSD:
    def __init__(self, templates):
        sld_route_images = np.lib.stride_tricks.sliding_window_view(templates, window_shape=(32, 16, 3), axis=(1, 2, 3))[:, 0, :, 0]
        self.norm_sld_route_images = sld_route_images/sld_route_images.mean(axis=(2,3,4))[:,:,np.newaxis, np.newaxis, np.newaxis]

    def ssd(self, image):
        return ((image - self.norm_sld_route_images)**2).mean(axis=(2,3,4))


class LostColorEdge:
    def __init__(self):
        self.preds_r = self.preds_g = self.preds_b = None
        self.mag_r = self.mag_g = self.mag_b = None

    def lost_q(self, template, image):
        sobel_x_template = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y_template = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize=5)
        sobel_x_image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y_image = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        mag_template_r = np.linalg.norm(np.array([sobel_x_template[:, :, 0], sobel_y_template[:, :, 0]]), axis=0)
        mag_template_g = np.linalg.norm(np.array([sobel_x_template[:, :, 1], sobel_y_template[:, :, 1]]), axis=0)
        mag_template_b = np.linalg.norm(np.array([sobel_x_template[:, :, 2], sobel_y_template[:, :, 2]]), axis=0)
        mag_image_r = np.linalg.norm(np.array([sobel_x_image[:, :, 0], sobel_y_image[:, :, 0]]), axis=0)
        mag_image_g = np.linalg.norm(np.array([sobel_x_image[:, :, 1], sobel_y_image[:, :, 1]]), axis=0)
        mag_image_b = np.linalg.norm(np.array([sobel_x_image[:, :, 2], sobel_y_image[:, :, 2]]), axis=0)

        dir_template = np.arctan2(sobel_x_template, sobel_y_template)
        dir_image = np.arctan2(sobel_x_image, sobel_y_image)

        m_r = (1 - np.exp(-mag_template_r/40))*4
        m_g = (1 - np.exp(-mag_template_g/40))*4
        m_b = (1 - np.exp(-mag_template_b/40))*4

        self.preds_r = vonmises(m_r, dir_template[:,:,0]).logpdf(dir_image[:,:,0]) - np.log(1.0/(2*np.pi))
        self.preds_g = vonmises(m_g, dir_template[:,:,1]).logpdf(dir_image[:,:,1]) - np.log(1.0 / (2 * np.pi))
        self.preds_b = vonmises(m_b, dir_template[:,:,2]).logpdf(dir_image[:,:,2]) - np.log(1.0 / (2 * np.pi))

        self.mag_r = chi2(mag_template_r+1).logpdf(mag_image_r+1)/20
        self.mag_g = chi2(mag_template_g+1).logpdf(mag_image_g+1)/20
        self.mag_b = chi2(mag_template_b+1).logpdf(mag_image_b+1)/20

        return self.preds_r.sum() + self.preds_g.sum() + self.preds_b.sum() + self.mag_r.sum() + self.mag_g.sum() + self.mag_b.sum()

    def debug(self):
        preds = self.preds_r + self.preds_g + self.preds_b
        resized_angle_error = cv2.resize(preds, (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_mag_r_error = cv2.resize(self.mag_r, (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_mag_g_error = cv2.resize(self.mag_g, (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_mag_b_error = cv2.resize(self.mag_b, (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_mag_error = resized_mag_r_error + resized_mag_g_error + resized_mag_b_error
        debug_image = np.zeros([256, 513, 3])
        debug_image[:, :256, 2] = np.clip(resized_angle_error/2.0, 0.0, 1.0)
        debug_image[:, :256, 0] = np.clip(-resized_angle_error / 2.0, 0.0, 1.0)
        debug_image[:, 257:, 0] = np.clip(-resized_mag_error, 0.0, 1.0)
        debug_image[:, 257:, 2] = np.clip(resized_mag_error, 0.0, 1.0)
        return debug_image


class CatNav(Node):
    def __init__(self):
        super().__init__("ant_nav_1")
        self.declare_parameter('route_folder', './default_route_folder')
        self.declare_parameter('route_loop', False)
        self.declare_parameter('lost_edge_threshold', 150.0)
        self.declare_parameter('drive', True)
        self.declare_parameter('lost_seq_len', 5)
        self.declare_parameter('warning_time', .25)
        self.declare_parameter("publish_diagnostic", True)
        self.declare_parameter('angle_ratio', 36.)
        self.declare_parameter('stop_on_last', 5)
        self.route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.route_loop = self.get_parameter('route_loop').get_parameter_value().bool_value
        self.lost_edge_threshold = self.get_parameter('lost_edge_threshold').get_parameter_value().double_value
        self.drive = self.get_parameter('drive').get_parameter_value().bool_value
        self.lost_seq_len  = self.get_parameter('lost_seq_len').get_parameter_value().integer_value
        self.warning_time = self.get_parameter('warning_time').get_parameter_value().double_value
        self.angle_ratio = self.get_parameter('angle_ratio').get_parameter_value().double_value
        self.stop_on_last = self.get_parameter('stop_on_last').get_parameter_value().integer_value
        self.blur = 1
        self.route_images = self.load_images()
        self.last_image_idx = self.route_images.shape[0]-1
        self.image_idx = 0
        self.lost = self.lost_seq_len
        self.ssd = SSD(self.route_images)
        self.image_subscription = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            1)
        self.twist_publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        if self.get_parameter('publish_diagnostic').get_parameter_value().bool_value:
            self.diagnostic_image_publisher = self.create_publisher(Image, "/diagnostic_image", 10)
        else:
            self.diagnostic_image_publisher = None
        self.bridge = CvBridge()
        self.lostObj = LostColorEdge()
        print("Initialized.")

    def normalize(self, image):
        """Binarizes onto (-1,1) using median."""
        return image/image.mean()

    def load_images(self):
        """Reads in images resizes to 64x64."""
        files = glob.glob(f"{self.route_folder}/*.jpg")
        files.sort()
        resized = np.array([np.array(PILImage.open(fname).resize((32,32))).astype(np.float32)/256. for fname in files])
        filtered = gaussian_filter(resized, sigma=(0, self.blur, self.blur, 0))
        return filtered

    def publish_twist(self, header, speed, angular_velocity):
        twist_stamped = TwistStamped()
        twist_stamped.header = header
        twist_stamped.twist.linear.x = speed
        twist_stamped.twist.angular.z = angular_velocity
        self.twist_publisher.publish(twist_stamped)

    def debug_image(self, image, template):
        top = np.zeros([256, 513, 3])
        resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_template = cv2.resize(template, (256, 256), interpolation=cv2.INTER_NEAREST)

        top[:256,:256 ] = resized_image
        top[:256,257:] = resized_template
        lost_debug_image = self.lostObj.debug()
        canvas = cv2.vconcat([top, lost_debug_image])
        cv2.line(canvas, (256, 0), (256, 512), color=(0, 1, 0))
        cv2.line(canvas, (4*16, 0), (4*16, 256), color=(1,0,0))
        cv2.line(canvas, (256 + 4 * 16, 0), (256 + 4 * 16, 256), color=(1, 0, 0))
        cv2.line(canvas, (0, 4*8), (512, 4*8), color=(1, 0, 0))
        return canvas

    def get_drive_instructions(self, image):
        image_diffs = self.ssd.ssd(image)
        template_min = np.sqrt(image_diffs.min())
        image_idx, angle = np.unravel_index(np.argmin(image_diffs, axis=None), image_diffs.shape)
        angle = np.argmin(image_diffs[(image_idx + 1) % self.last_image_idx])
        sub_window_idx = np.argmin(image_diffs[image_idx])
        return image_idx, sub_window_idx, angle-8, template_min

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
        image = np.array(pil_image.resize((32,32))).astype(np.float32)/256.
        smoothed_image = gaussian_filter(image, sigma=(self.blur, self.blur, 0))
        centre_image = smoothed_image[:, 8:24]
        norm_image = self.normalize(centre_image).astype(np.float32)
        image_idx, sub_window_idx, angle, template_min = self.get_drive_instructions(norm_image)

        lost_edge_min = self.lostObj.lost_q(self.ssd.norm_sld_route_images[image_idx, sub_window_idx], norm_image)
        print(f'matched image idx {image_idx}, angle={angle}, template_min={template_min:.2f}, edge_min={lost_edge_min:.2f}')
        if lost_edge_min < self.lost_edge_threshold:
            self.lost += 1
        else:
            self.lost = 0
        if self.drive and self.lost < self.lost_seq_len and (image_idx < self.last_image_idx-self.stop_on_last or self.route_loop):
            speed = 0.05
            angular_velocity = angle/self.angle_ratio
            self.publish_twist(image_msg.header, speed, angular_velocity)
        if self.diagnostic_image_publisher:
            debug_image = self.debug_image(centre_image, self.route_images[image_idx, :, sub_window_idx:sub_window_idx + 16])
            debug_image_msg = self.bridge.cv2_to_imgmsg((debug_image*255).astype(np.uint8),
                                                       encoding="rgb8")
            self.diagnostic_image_publisher.publish(debug_image_msg)
        self.warnings(image_msg.header.stamp, time_received)


rclpy.init()
ant_nav = CatNav()
rclpy.spin(ant_nav)
ant_nav.destroy_node()
rclpy.shutdown()
