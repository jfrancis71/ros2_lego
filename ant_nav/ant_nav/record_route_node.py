import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class RouteRecorder(Node):
    def __init__(self):
        super().__init__("route_recorder")
        self.declare_parameter('route_folder', './default_route_folder')
        self.declare_parameter('record_interval', .05)
        self.route_folder = self.get_parameter('route_folder').get_parameter_value().string_value
        self.record_interval = self.get_parameter('record_interval').get_parameter_value().double_value
        self.image_subscription = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            1)
        self.pose_subscription = self.create_subscription(
            Odometry,
            "/odom",
            self.odometry_callback,
            10)
        self.bridge = CvBridge()
        self.image_idx = 0
        self.last_image = None
        self.last_inertial_position = None
        self.last_inertial_orientation = None
        os.makedirs(self.route_folder, exist_ok=True)
        print("Initialized.")

    def moved(self, current_inertial_position, current_inertial_orientation, last_inertial_position, last_inertial_orientation):
        if np.linalg.norm(current_inertial_position - last_inertial_position) > self.record_interval:
            return True
        else:
            return False

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        im = PILImage.fromarray(cv_image)
        self.last_image = im

    def save_image(self):
        self.last_image.save(f"{self.route_folder}/{self.image_idx:04d}.jpg")
        self.recorded_image = self.last_image
        print("Saving image ", self.image_idx)
        self.image_idx += 1

    def odometry_callback(self, msg):
        current_inertial_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        current_inertial_orientation = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
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
route_recorder = RouteRecorder()
rclpy.spin(route_recorder)
route_recorder.destroy_node()
rclpy.shutdown()
