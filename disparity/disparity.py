import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import Image
import numpy as np
import cv2 as cv


class DisparityNode(Node):
    def __init__(self):
        super().__init__("dispa_node")
        self.bridge = CvBridge()
        queue_size = 1
        self.disparity_image_sub = Subscriber(self, DisparityImage, "/disparity")
        self.left_image_sub = Subscriber(self, Image, "/left/image_rect_color")
        self.right_image_sub = Subscriber(self, Image, "/right/image_rect_color")
        self.sync = TimeSynchronizer([self.left_image_sub, self.right_image_sub, self.disparity_image_sub], queue_size)
        self.sync.registerCallback(self.disparity_cb)
        self.annotated_image_publisher = \
                self.create_publisher(Image, "annotated_image", 10)

    def disparity_cb(self, left_img_msg, right_img_msg, disparity_msg):
        disparity_array = self.bridge.imgmsg_to_cv2(disparity_msg.image, desired_encoding='32FC1')
        left_cv_image = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding="rgb8")
        left_image = left_cv_image.copy().transpose((2, 0, 1))
        print("LEFT=", left_image.shape)
        right_cv_image = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding="rgb8")
        right_image = right_cv_image.copy().transpose((2, 0, 1))
        print("R=", right_image.dtype)
        image = np.zeros([3,40,320], dtype=np.uint8)
        image[:,20:] = left_image[:,110:130]
        image[:,:20] = right_image[:,110:130]
        my = image.transpose(1, 2, 0).copy()
        print("DISP=", disparity_array[120])
        for i in range(0, 320, 10):
            disp = int(disparity_array[120,i])
            if disp != -1:
                cv.line(my, (i,10),(i+disp,30),(255,0,0),2)
        ros2_image_msg = self.bridge.cv2_to_imgmsg(my,
                                                   encoding="rgb8")

        ros2_image_msg.header = disparity_msg.header
        self.annotated_image_publisher.publish(ros2_image_msg)

        
        #print(disparity_array[120])


rclpy.init()
disparity_node = DisparityNode()
rclpy.spin(disparity_node)
rclpy.shutdown()

