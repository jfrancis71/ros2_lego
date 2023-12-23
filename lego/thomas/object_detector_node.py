from torchvision.models import detection
import numpy as np
import argparse
import torch
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import ObjectHypothesis
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__("object_detector")
        self.subscription = self.create_subscription(
            Image,
            "/image",
            self.listener_callback,
            10)
        self.detected_objects_publisher = self.create_publisher(Detection2DArray, "detected_objects", 10)
        self.detected_objects_image_publisher = self.create_publisher(Image, "detected_objects_image", 10)
        self.bridge = CvBridge()
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('detection_threshold', 0.9)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.detection_threshold = self.get_parameter('detection_threshold').get_parameter_value().double_value
        self.model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1", progress=True, weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1").to(self.device)
        self.class_labels = detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta["categories"]
        self.model.eval()

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = cv_image.copy().transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.to(self.device)
        detections = self.model(image)[0]
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        for label_id, score, box in zip(detections["labels"], detections["scores"], detections["boxes"]):
            if score < self.detection_threshold:
                continue
            box = box.detach().type(torch.int64)
            cv2.rectangle(cv_image, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (255, 255, 0), 4)
            detection = Detection2D()
            detection.header = msg.header
            object_hypothesis_with_pose = ObjectHypothesisWithPose()
            object_hypothesis = ObjectHypothesis()
            object_hypothesis.class_id = self.class_labels[label_id]
            object_hypothesis.score = score.detach().item()
            object_hypothesis_with_pose.hypothesis = object_hypothesis
            detection.results.append(object_hypothesis_with_pose)
            bounding_box = BoundingBox2D()
            bounding_box.center.position.x = float((box[0] + box[2])/2)
            bounding_box.center.position.y = float((box[1] + box[3])/2)
            bounding_box.center.theta = 0.0
            bounding_box.size_x = float(2*(bounding_box.center.position.x - box[0]))
            bounding_box.size_y = float(2*(bounding_box.center.position.y - box[1]))
            detection.bbox = bounding_box
            detection_array.detections.append(detection)
        self.detected_objects_publisher.publish(detection_array)
        ros2_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
        ros2_image_msg.header = msg.header
        self.detected_objects_image_publisher.publish(ros2_image_msg)


rclpy.init()
object_detector_node = ObjectDetectorNode()
rclpy.spin(object_detector_node)
object_detector_node.destroy_node()
rclpy.shutdown()
