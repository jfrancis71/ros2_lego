# This uses CameraInfoManager, works with https://github.com/furbrain/image_common
# You might to selectively build just the camera_info_manager_py package (from above).

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from sensor_msgs.srv import SetCameraInfo
from camera_info_manager import CameraInfoManager


class StereoSplitNode(Node):
    def __init__(self):
        super().__init__("stereo_split_node")
        self.declare_parameter('left_camera_info_url', '')
        self.declare_parameter('right_camera_info_url', '')
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
        self.left_ci = CameraInfoManager(self,
            url=self.get_parameter('left_camera_info_url'). \
                get_parameter_value().string_value,
            namespace="/stereo/left")
        self.right_ci = CameraInfoManager(self,
            url=self.get_parameter('right_camera_info_url'). \
                get_parameter_value().string_value,
            namespace="/stereo/right")
        self.left_ci.loadCameraInfo()
        self.right_ci.loadCameraInfo()

    def image_callback(self, image_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        mono_image_width = cv_image.shape[1]//2
        left_ros2_image_msg = self.bridge.cv2_to_imgmsg(cv_image[:, :mono_image_width], encoding="bgr8")
        left_ros2_image_msg.header = image_msg.header
        self.left_image_publisher.publish(left_ros2_image_msg)
        right_ros2_image_msg = self.bridge.cv2_to_imgmsg(cv_image[:, mono_image_width:], encoding="bgr8")
        right_ros2_image_msg.header = image_msg.header
        self.right_image_publisher.publish(right_ros2_image_msg)
        if self.left_ci.camera_info is not None and self.right_ci.camera_info is not None:
            self.left_ci.camera_info.header = image_msg.header
            self.right_ci.camera_info.header = image_msg.header
            self.left_cam_info_publisher.publish(self.left_ci.camera_info)
            self.right_cam_info_publisher.publish(self.right_ci.camera_info)


rclpy.init()
stereo_node = StereoSplitNode()
rclpy.spin(stereo_node)
stereo_node.destroy_node()
rclpy.shutdown()
