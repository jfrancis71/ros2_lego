import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml


class StereoSplitNode(Node):
    def __init__(self):
        super().__init__("stereo_split_node")
        self.subscription = self.create_subscription(
            Image,
            "/image",
            self.image_callback,
            1)
        self.left_image_publisher = \
                self.create_publisher(Image, "/stereo/left/image_raw", 1)
        self.right_image_publisher = \
                self.create_publisher(Image, "/stereo/right/image_raw", 1)
        self.left_cam_info_publisher = \
                self.create_publisher(CameraInfo, "/stereo/left/camera_info", 1)
        self.right_cam_info_publisher = \
                self.create_publisher(CameraInfo, "/stereo/right/camera_info", 1)
        self.bridge = CvBridge()
        try:
            with open("/root/ros2_ws/calib/left.yaml", 'r') as stream:
                left_cam_data = yaml.safe_load(stream)
                self.left_cam_info = CameraInfo()
                self.left_cam_info.width = 320
                self.left_cam_info.height = 240
                self.left_cam_info.distortion_model = 'plumb_bob'
                self.left_cam_info.d = left_cam_data["distortion_coefficients"]["data"]
                self.left_cam_info.r = left_cam_data["rectification_matrix"]["data"]
                self.left_cam_info.p = left_cam_data["projection_matrix"]["data"]
                self.left_cam_info.k = left_cam_data["camera_matrix"]["data"]
        except FileNotFoundError:
            self.left_cam_info = None
        try:
            with open("/root/ros2_ws/calib/right.yaml", 'r') as stream:
                right_cam_data = yaml.safe_load(stream)
                self.right_cam_info = CameraInfo()
                self.right_cam_info.width = 320
                self.right_cam_info.height = 240
                self.right_cam_info.distortion_model = 'plumb_bob'
                self.right_cam_info.d = right_cam_data["distortion_coefficients"]["data"]
                self.right_cam_info.r = right_cam_data["rectification_matrix"]["data"]
                self.right_cam_info.p = right_cam_data["projection_matrix"]["data"]
                self.right_cam_info.k = right_cam_data["camera_matrix"]["data"]
        except FileNotFoundError:
            self.right_cam_info = None

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        left_ros2_image_msg = self.bridge.cv2_to_imgmsg(cv_image[:, :320], encoding="bgr8")
        left_ros2_image_msg.header = image_msg.header
        self.left_image_publisher.publish(left_ros2_image_msg)
        right_ros2_image_msg = self.bridge.cv2_to_imgmsg(cv_image[:, 320:], encoding="bgr8")
        right_ros2_image_msg.header = image_msg.header
        self.right_image_publisher.publish(right_ros2_image_msg)
        if self.left_cam_info is not None and self.right_cam_info is not None:
            self.left_cam_info.header = image_msg.header
            self.right_cam_info.header = image_msg.header
            self.left_cam_info_publisher.publish(self.left_cam)
            self.right_cam_info_publisher.publish(self.right_cam)


rclpy.init()
stereo_node = StereoSplitNode()
rclpy.spin(stereo_node)
stereo_node.destroy_node()
rclpy.shutdown()
