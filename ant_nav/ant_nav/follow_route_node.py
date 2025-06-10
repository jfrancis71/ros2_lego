import glob
import numpy as np
from scipy.ndimage import gaussian_filter
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
        self.heading_text = np.zeros([30, 513, 3]).astype(np.float32)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.heading_text, 'Edge Orientation Error', (60, 20), font, .5, (1, 1, 1), 1, cv2.LINE_AA)
        cv2.putText(self.heading_text, 'Edge Magnitude Error', (280, 20), font, .5, (1, 1, 1), 1, cv2.LINE_AA)

    def prediction_error(self, offset, template, image):
        template_centre_image = template[:, offset: offset + 16]
        template_norm = template_centre_image/template.mean()
        centre_image = image[:, 8:24].astype(np.float32)
        image_norm = centre_image/centre_image.mean()
        sobel_x_template = cv2.Sobel(template_norm, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y_template = cv2.Sobel(template_norm, cv2.CV_64F, 0, 1, ksize=5)
        sobel_x_image = cv2.Sobel(image_norm, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y_image = cv2.Sobel(image_norm, cv2.CV_64F, 0, 1, ksize=5)
        mag_template = np.linalg.norm(np.array([sobel_x_template, sobel_y_template]), axis=0)
        mag_image = np.linalg.norm(np.array([sobel_x_image, sobel_y_image]), axis=0)
        dir_template = np.arctan2(sobel_x_template, sobel_y_template)
        dir_image = np.arctan2(sobel_x_image, sobel_y_image)
        m = (1 - np.exp(-mag_template/40))*4
        edge_direction_error = vonmises(m, dir_template).logpdf(dir_image) - np.log(1.0/(2*np.pi))
        edge_mag_error = (chi2(mag_template+1).logpdf(mag_image+1) - chi2(10).logpdf(mag_image+1))/20
        return edge_direction_error, edge_mag_error

    def lost_error(self, edge_direction_error, edge_mag_error):
        return edge_direction_error.sum() + edge_mag_error.sum()

    def diagnostic_image(self, offset, col_edge_direction_error, col_edge_mag_error):
        edge_direction_error = col_edge_direction_error.sum(axis=-1)
        edge_mag_error = col_edge_mag_error.sum(axis=-1)
        resized_angle_error = cv2.resize(edge_direction_error, (128, 256), interpolation=cv2.INTER_NEAREST)
        resized_mag_error = cv2.resize(edge_mag_error, (128, 256), interpolation=cv2.INTER_NEAREST)
        diagnostic_angle_error = np.zeros([256, 256, 3]).astype(np.float32)
        scale = int(256/32)
        diagnostic_angle_error[:, offset*scale:(offset+16)*scale, 2] = np.clip(resized_angle_error/2.0, 0.0, 1.0)
        diagnostic_angle_error[:, offset*scale:(offset+16)*scale, 0] = np.clip(-resized_angle_error / 2.0, 0.0, 1.0)
        diagnostic_mag_error = np.zeros([256, 256, 3]).astype(np.float32)
        diagnostic_mag_error[:, offset*scale:(offset+16)*scale, 2] = np.clip(resized_mag_error, 0.0, 1.0)
        diagnostic_mag_error[:, offset*scale:(offset+16)*scale, 0] = np.clip(-resized_mag_error, 0.0, 1.0)
        cv2.line(diagnostic_angle_error, (8*16, 0), (8*16, 256), color=(0, .7, .7))
        cv2.line(diagnostic_mag_error, (8 * 16, 0), (8 * 16, 256), color=(0, .7, .7))
        divider = np.zeros([256, 1, 3]).astype(np.float32)
        divider[:, :, 1] = 1.0
        diagnostic_image = cv2.hconcat([diagnostic_angle_error, divider, diagnostic_mag_error])
        cv2.line(diagnostic_image, (0, 16 * 8), (512, 16 * 8), color=(0, .7, .7))
        canvas = cv2.vconcat([self.heading_text, diagnostic_image])
        return canvas


class Localizer:
    def __init__(self, route_images):
        self.blur = 1
        filtered = gaussian_filter(route_images, sigma=(0, self.blur, self.blur, 0))
        sld_route_images = np.lib.stride_tricks.sliding_window_view(filtered, window_shape=(32, 16, 3), axis=(1, 2, 3))[:, 0, :, 0]
        self.norm_sld_route_images = sld_route_images/sld_route_images.mean(axis=(2,3,4))[:,:,np.newaxis, np.newaxis, np.newaxis]
        self.last_image_idx = route_images.shape[0]-1
        self.heading_text = np.zeros([30, 513, 3]).astype(np.float32)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.heading_text, 'Current Camera Image', (60, 20), font, .5, (1, 1, 1), 1, cv2.LINE_AA)
        cv2.putText(self.heading_text, 'Best Template Match', (280, 20), font, .5, (1, 1, 1), 1, cv2.LINE_AA)

    def localize(self, image):
        centre_image = image[:, 8:24].astype(np.float32)
        smoothed_image = gaussian_filter(centre_image, sigma=(self.blur, self.blur, 0))
        norm_image = smoothed_image/smoothed_image.mean()
        image_diffs = ((norm_image - self.norm_sld_route_images)**2).mean(axis=(2,3,4))
        template_min = np.sqrt(image_diffs.min())
        image_idx, _ = np.unravel_index(np.argmin(image_diffs, axis=None), image_diffs.shape)
        offset = np.argmin(image_diffs[image_idx])
        next_offset = np.argmin(image_diffs[(image_idx + 1) % self.last_image_idx])
        return image_idx, offset, next_offset, template_min

    def diagnostic_image(self, offset, image, template):
        resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_template = cv2.resize(template, (256, 256), interpolation=cv2.INTER_NEAREST)
        divider = np.zeros([256, 1, 3]).astype(np.float32)
        divider[:, :, 1] = 1.0
        cv2.line(resized_image, (64, 0), (64, 256), color=(0, .7, .7))
        cv2.line(resized_image, (64+128, 0), (64+128, 256), color=(0, .7, .7))
        scale = int(256/32)
        cv2.line(resized_template, (offset * scale, 0), (offset * scale, 256), color=(0, .7, .7))
        cv2.line(resized_template, ((offset+16) * scale, 0), ((offset+16) * scale, 256), color=(0, .7, .7))
        diagnostic_image = cv2.hconcat([resized_image, divider, resized_template])
        cv2.line(diagnostic_image, (0, 16*8), (512, 16*8), color=(0, .7, .7))
        canvas = cv2.vconcat([self.heading_text, diagnostic_image])
        return canvas


def load_images(route_folder):
    files = glob.glob(f"{route_folder}/*.jpg")
    if files == []:
        raise RuntimeError(f'Error, no files in folder {route_folder}.')
    files.sort()
    resized = np.array([np.array(PILImage.open(fname).resize((32,32))).astype(np.float32)/256. for fname in files])
    return resized


class CatNav(Node):
    def __init__(self):
        super().__init__("cat_nav")
        self.declare_parameter('route_folder', './default_route_folder')
        self.declare_parameter('route_loop', False)
        self.declare_parameter('lost_edge_threshold', 450.0)
        self.declare_parameter('self_drive', True)
        self.declare_parameter('lost_seq_len', 5)
        self.declare_parameter('warning_time', .25)
        self.declare_parameter("publish_diagnostic", True)
        self.declare_parameter('angle_ratio', 36.)
        self.declare_parameter('stop_on_last', 5)
        self.declare_parameter('forward_speed', .05)
        route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.route_loop = self.get_parameter('route_loop').get_parameter_value().bool_value
        self.lost_edge_threshold = self.get_parameter('lost_edge_threshold').get_parameter_value().double_value
        self.self_drive = self.get_parameter('self_drive').get_parameter_value().bool_value
        self.lost_seq_len  = self.get_parameter('lost_seq_len').get_parameter_value().integer_value
        self.warning_time = self.get_parameter('warning_time').get_parameter_value().double_value
        self.angle_ratio = self.get_parameter('angle_ratio').get_parameter_value().double_value
        self.stop_on_last = self.get_parameter('stop_on_last').get_parameter_value().integer_value
        self.forward_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        self.route_images = load_images(route_folder)
        self.last_image_idx = self.route_images.shape[0]-1
        # Let's start assuming lost; we'll reset this later in image_callback if good match found
        self.lost_counter = self.lost_seq_len
        self.image_subscription = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            1)
        self.twist_publisher = self.create_publisher(TwistStamped, "/cmd_vel", 1)
        if self.get_parameter('publish_diagnostic').get_parameter_value().bool_value:
            self.diagnostic_image_publisher = self.create_publisher(Image, "/diagnostic_image", 1)
        else:
            self.diagnostic_image_publisher = None
        self.bridge = CvBridge()
        self.localizer = Localizer(self.route_images)
        self.lost_detector = LostDetector()
        self.get_logger().info("Node has started.")

    def publish_twist(self, header, twist_linear_x, twist_angular_z):
        twist_stamped = TwistStamped()
        twist_stamped.header = header
        twist_stamped.twist.linear.x = twist_linear_x
        twist_stamped.twist.angular.z = twist_angular_z
        self.twist_publisher.publish(twist_stamped)

    def diagnostic_image(self, offset, image, template, edge_direction_error, edge_mag_error):
        top = self.localizer.diagnostic_image(offset, image, template)
        lost_diagnostic_image = self.lost_detector.diagnostic_image(offset, edge_direction_error, edge_mag_error)
        canvas = cv2.vconcat([top, lost_diagnostic_image])
        return canvas

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
        image_idx, offset, next_offset, template_min = self.localizer.localize(image)
        edge_direction_error, edge_mag_error = self.lost_detector.prediction_error(
            offset, self.route_images[image_idx], image)
        if self.diagnostic_image_publisher:
            diagnostic_image = self.diagnostic_image(offset, image,
                self.route_images[image_idx], edge_direction_error, edge_mag_error)
            diagnostic_image_msg = self.bridge.cv2_to_imgmsg((diagnostic_image*255).astype(np.uint8),
                                                       encoding="rgb8")
            self.diagnostic_image_publisher.publish(diagnostic_image_msg)
        lost_min = self.lost_detector.lost_error(edge_direction_error, edge_mag_error)
        info_msg = f'matched image idx {image_idx}, centered offset={offset-8}, template_min={template_min:.2f}, lost_edge={lost_min:.2f}'
        self.get_logger().info(info_msg)
        if lost_min < self.lost_edge_threshold:
            self.lost_counter += 1
        else:
            self.lost_counter = 0
        if self.self_drive and self.lost_counter < self.lost_seq_len and (image_idx < self.last_image_idx-self.stop_on_last or self.route_loop):
            twist_linear_x = self.forward_speed
            twist_angular_z = (next_offset-8)/self.angle_ratio
            self.publish_twist(image_msg.header, twist_linear_x, twist_angular_z)
        self.timing_warnings(image_msg.header.stamp, time_received)


rclpy.init()
cat_nav = CatNav()
rclpy.spin(cat_nav)
cat_nav.destroy_node()
rclpy.shutdown()
