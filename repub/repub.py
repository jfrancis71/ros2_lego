import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from sensor_msgs.srv import SetCameraInfo


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
        self.left_srv = self.create_service(SetCameraInfo, '/stereo/left_camera/set_camera_info', self.set_left_camera_info_callback)
        self.right_srv = self.create_service(SetCameraInfo, '/stereo/right_camera/set_camera_info', self.set_right_camera_info_callback)
        #self.left_srv = self.create_service(SetCameraInfo, 'left_camera/set_camera_info', self.set_left_camera_info_callback)
        #self.right_srv = self.create_service(SetCameraInfo, 'right_camera/set_camera_info', self.set_right_camera_info_callback)
        self.bridge = CvBridge()
        self.bridge = CvBridge()
        self.left_cam_info = None
        self.right_cam_info = None

    def set_left_camera_info_callback(self, request, response):
        print("Hello received left service request.")
        self.left_cam_info = request.camera_info
        response.success = True
        response.status_message = ""
        return response

    def set_right_camera_info_callback(self, request, response):
        print("Hello received right service request.")
        self.right_cam_info = request.camera_info
        response.success = True
        response.status_message = ""
        return response

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
            self.left_cam_info_publisher.publish(self.left_cam_info)
            self.right_cam_info_publisher.publish(self.right_cam_info)


rclpy.init()
stereo_node = StereoSplitNode()
rclpy.spin(stereo_node)
stereo_node.destroy_node()
rclpy.shutdown()
