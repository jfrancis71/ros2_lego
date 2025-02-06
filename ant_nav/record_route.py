import rclpy
from PIL import Image as PILImage
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from tf_transformations import euler_from_quaternion


class AntNav(Node):
    def __init__(self):
        super().__init__("ant_nav")
        self.image_subscription = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            10)
        self.pose_subscription = self.create_subscription(
            Odometry,
            "/omni_wheel_controller/odom",
            self.odometry_callback,
            10)
        self.bridge = CvBridge()
        self.declare_parameter('route_folder', './default_route_folder')
        self.route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.image_idx = 0
        self.last_image = None
        self.last_inertial_position = None
        self.last_inertial_orientation = None

    def normalize(self, image):
        """Binarizes onto (-1,1) using median."""
        return ((image - np.median(image, axis=(0,1)))>0)*1.0

    def moved(self, current_inertial_position, current_inertial_orientation, last_inertial_position, last_inertial_orientation):
        if np.linalg.norm(current_inertial_position - last_inertial_position) > .05:
            return True
        else:
            return False

    def image_changed(self):
        if ((self.normalize(self.last_image)-self.normalize(self.recorded_image))**2).mean() > .2:
            return True
        else:
            return False

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        im = PILImage.fromarray(cv_image)
        self.last_image = im

    def save_image(self):
        self.image_idx += 1
        self.last_image.save(f"{self.route_folder}/{self.image_idx:04d}.jpg")
        self.recorded_image = self.last_image
        print("Saving image", self.image_idx)

    def odometry_callback(self, msg):
        print("POSE", msg.pose.pose.position)
        current_inertial_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        current_inertial_orientation = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        print("Current IO=", current_inertial_orientation)

        if self.last_inertial_position is None:
            self.last_inertial_position = current_inertial_position
            self.last_inertial_orientation = current_inertial_orientation
            if self.last_image is not None:
                self.save_image()

        if self.moved(current_inertial_position, current_inertial_orientation, self.last_inertial_position, self.last_inertial_orientation):
            self.last_inertial_position = current_inertial_position
            self.last_inertial_orientation = current_inertial_orientation
            self.save_image()


rclpy.init()
ant_nav = AntNav()
rclpy.spin(ant_nav)
ant_nav.destroy_node()
rclpy.shutdown()
