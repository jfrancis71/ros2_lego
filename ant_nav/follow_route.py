import rclpy
from PIL import Image as PILImage
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import glob
from scipy.signal import correlate


class AntNav1(Node):
    def __init__(self):
        super().__init__("ant_nav_1")
        self.image_subscription = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.image_publisher = self.create_publisher(Image, "/debug_image", 10)
        self.bridge = CvBridge()
        self.declare_parameter('route_folder', './default_route_folder')
        self.route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.images = self.load_images()
        self.last_image_idx = self.images.shape[0]-1

    def normalize(self, image):
        """Binarizes onto (-1,1) using median."""
        return ((image - np.median(image, axis=(0,1)))>0)*1.0

    def load_images(self):
        """Reads in images resizes to 64x64. Takes subslices of 32x64 and normalizes"""
        files = glob.glob(f"{self.route_folder}/*.jpg")
        files.sort()
        self.resized = np.array([np.array(PILImage.open(fname).resize((64,64)))/256. for fname in files])
        normalized = np.array([np.array([self.normalize(self.resized[image_idx, :, offset:32+offset]) for offset in range(32)]) for image_idx in range(len(files))])
        return normalized

    def route_image_diff(self, image):
#        return np.array([correlate(image, self.images[idx], mode='valid')/(64*64*3) for idx in range(self.images.shape[0])])[:,0,:,0]
#        return ((self.images - image)**2).mean(axis=(1,2,3))
        centre_image = image[:, 16:48]
        norm_image = self.normalize(centre_image)
        diffs = ((norm_image - self.images)**2).mean(axis=(2,3,4))
        return diffs

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        image = np.array(PILImage.fromarray(cv_image).resize((64,64)))/256.
#        image = self.resized[0]
        image_diffs = self.route_image_diff(image)
        twist = Twist()
        debug_image_msg = self.bridge.cv2_to_imgmsg((image_diffs.clip(0.0, 1.0)*256).astype(np.int8), "8SC1")
        cmin = image_diffs.min()
        image_idx, angle = np.unravel_index(np.argmin(image_diffs, axis=None), image_diffs.shape)
        print("image_idx:", image_idx, ", angle: ", angle)
        if image_idx != self.last_image_idx:
            angle = np.argmin(image_diffs[image_idx+1])
        if image_idx == self.last_image_idx or cmin > 1.5:
            twist.linear.x = 0.00
            twist.angular.z = 0.00
        else:
            twist.linear.x = 0.05
#            twist.angular.z = 0.00
#            if angle > 4:
#                twist.angular.z = -0.5
#            if angle < 4:
#                twist.angular.z = 0.5
            twist.angular.z = (angle-16)/48
        self.publisher.publish(twist)
        self.image_publisher.publish(debug_image_msg)


rclpy.init()
ant_nav = AntNav1()
rclpy.spin(ant_nav)
ant_nav.destroy_node()
rclpy.shutdown()
