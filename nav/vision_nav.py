import collections
import itertools
import math
import numpy as np
import torch
from cv_bridge import CvBridge
from torchvision.models import detection as detection_model
from torchvision.utils import draw_bounding_boxes
from vision_msgs.msg import BoundingBox2D, ObjectHypothesis, ObjectHypothesisWithPose
from vision_msgs.msg import Detection2D, Detection2DArray
from rclpy.node import Node
from sensor_msgs.msg import Image




f_x = 494
f_y = 294
WorldPoint = collections.namedtuple("WorldPoint", "x, y, z")
WorldObject = collections.namedtuple("WorldObject", "name, centre, bottom_left, bottom_right, top_left, top_right")
CameraPoint = collections.namedtuple("CameraPoint", "x, y")
BoundingBoxes = collections.namedtuple("BoundingBoxes", "center_x, center_y, width, height")  # Grid of bounding boxes
Detections = collections.namedtuple("Detections", "labels, bounding_boxes, detection_probabilities")

class VisionNav(Node):
    def __init__(self, num_grid_cells, num_orientation_cells):
        super().__init__("vision_nav_node")
        self.world_dog = WorldObject("dog",
            WorldPoint(1.5, 0.0, 0.27),
            WorldPoint(1.5, 0.11, .02),
            WorldPoint(1.5, -0.11, .02),
            WorldPoint(1.5, 0.11, .52),
            WorldPoint(1.5, -0.11, .52)
        )
        self.world_cat = WorldObject("cat",
                                     WorldPoint(.5, -1.5, 0.27),
                                     WorldPoint(.5-.11, 1.5, .02),
                                     WorldPoint(.5+.11, 1.5, .02),
                                     WorldPoint(.5-.11, 1.5, .52),
                                     WorldPoint(.5+.11, 1.5, .52)
                                     )
        self.num_grid_cells = num_grid_cells
        self.num_orientation_cells = num_orientation_cells

        self.world_grid_length = 3.0
        self.world_cell_size = self.world_grid_length/self.num_grid_cells
        self.world_z = .22

        self.world_position = (torch.arange(self.num_grid_cells)+0.5) * self.world_cell_size - self.world_grid_length/2.0
        self.world_xs = self.world_position
        self.world_ys = torch.flip(self.world_position, dims=[0])
        self.world_thetas = torch.arange(self.num_orientation_cells)*2*math.pi/self.num_orientation_cells
        self.world_x, self.world_y, self.world_theta = torch.meshgrid(self.world_xs, self.world_ys, self.world_thetas, indexing='xy')

        world_dog_boxes = self.world_to_bounding_boxes(self.world_dog)
        world_cat_boxes = self.world_to_bounding_boxes(self.world_cat)
        self.object_dictionary = {
            "dog": Detections(None, world_dog_boxes, self.box_probability(world_dog_boxes)),
            "cat": Detections(None, world_cat_boxes, self.box_probability(world_cat_boxes))}
        self.object_list = list(self.object_dictionary.keys())
        self.bridge = CvBridge()
        self.model = detection_model.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights="FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1",
            progress=True,
            weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1")
        self.class_labels = \
            detection_model.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta["categories"]
        self.model.eval()
        self.annotated_image_publisher = \
            self.create_publisher(Image, "annotated_image", 10)

    def prob_map(self, pose_boxes, detected_box):
        scale = 25.0
        res1 = torch.distributions.normal.Normal(pose_boxes.center_x, scale).log_prob(torch.tensor(detected_box.center_x))
        res2 = torch.distributions.normal.Normal(pose_boxes.width, scale).log_prob(torch.tensor(detected_box.width))
        res3 = torch.distributions.normal.Normal(pose_boxes.height, scale).log_prob(torch.tensor(detected_box.height))
        res4 = torch.distributions.normal.Normal(pose_boxes.center_y, scale).log_prob(
            torch.tensor(detected_box.center_y))
        res = res1 + res2 + res3 + res4
        mynorm = res # - torch.logsumexp(res, dim=[0, 1, 2], keepdim=False)
        smyprobs = torch.exp(mynorm)
        return smyprobs

    def probmessage_cond_a(self, detections_msg, proposals):
        # detections is Detections list, proposals is list of world_boxes
        # loop through proposals assign to a detection and recurse
        prob_dist_random = 0.05 * 0.01 * 0.01 * 0.01 * 0.01
        prob_dist_random_boxes = prob_dist_random+(self.world_x*0.0)
        if proposals == []:
            return prob_dist_random_boxes**len(detections_msg.labels)
        proposal, *remaining_proposals = proposals
        cum_prob = self.world_x * 0.0
        for assign_idx in range(len(detections_msg.labels)):
            rem_detections = detections_msg[:assign_idx] + detections_msg[assign_idx+1:]
            rem_detections = Detections(
                detections_msg[0][:assign_idx] + detections_msg[0][assign_idx+1:],
                detections_msg[1][:assign_idx] + detections_msg[1][assign_idx + 1:],
                detections_msg[2][:assign_idx] + detections_msg[2][assign_idx + 1:]
                                        )
            proposal_name = self.object_list[proposal]
            prob_assignment = self.prob_map(self.object_dictionary[proposal_name].bounding_boxes, BoundingBoxes(
                                            detections_msg.bounding_boxes.center_x[assign_idx],
                                            detections_msg.bounding_boxes.center_y[assign_idx],
                                            detections_msg.bounding_boxes.width[assign_idx],
                                            detections_msg.bounding_boxes.height[assign_idx]) )
            if proposal_name != detections_msg.labels[assign_idx]:
                prob_assignment = prob_assignment * 0.0
            rem_prob = self.probmessage_cond_a(rem_detections, remaining_proposals)
            total_prob = prob_assignment * rem_prob
            cum_prob += total_prob / len(detections_msg)
        return cum_prob

    def probmessage(self, image, header):
        detections, annotated_image = self.detect_image(image, header)
        comb = list(itertools.product([False,True], repeat=2))
        s = self.world_x * 0.0
        cond_assignment_probs = torch.zeros(len(comb), self.num_grid_cells, self.num_grid_cells, self.num_orientation_cells)
        assignment_probs = torch.zeros(len(comb), self.num_grid_cells, self.num_grid_cells, self.num_orientation_cells)
        for assignment_idx in range(len(comb)):
            probs = self.world_x * 0.0 + 1.0
            for idx in range(2):
                if comb[assignment_idx][idx] == False:
                    probs = probs * (1.0-self.object_dictionary[self.object_list[idx]].detection_probabilities)
                else:
                    probs = probs * self.object_dictionary[self.object_list[idx]].detection_probabilities
            assignments = [i for i, x in enumerate(comb[assignment_idx]) if x]
            assignment_probs[assignment_idx] = probs
            cond_assignment_probs[assignment_idx] = self.probmessage_cond_a(detections, assignments)
        joint_probs = cond_assignment_probs * assignment_probs
        tot_probs = joint_probs.sum(axis=0)
        return tot_probs, annotated_image

    def box_probability(self, boxes):
        """Computes the probability of a bounding box being detected.
        Example: If the bounding box is completely outside the camera field of view, it won't be detected.
        """
        cons_camera_left_x = torch.clip(boxes.center_x - boxes.width, min=-160, max=+160)
        cons_camera_right_x = torch.clip(boxes.center_x + boxes.width, min=-160, max=+160)
        cons_camera_bottom_y = torch.clip(boxes.center_y - boxes.height, min=-120, max=+120)
        cons_camera_top_y = torch.clip(boxes.center_y + boxes.height, min=-120, max=+120)
        cons_area = (cons_camera_right_x-cons_camera_left_x)*(cons_camera_top_y-cons_camera_bottom_y)
        area_ratio = cons_area / ((boxes.width*boxes.height)+cons_area)
        area_ratio = torch.nan_to_num(area_ratio, nan=0.0)
        return 0.05 * (1-area_ratio) + 0.95 * area_ratio

    def world_to_camera(self, world_point):
        world_translate_x = world_point.x - self.world_x
        world_translate_y = world_point.y - self.world_y
        world_translate_z = world_point.z - self.world_z
        world_rotate_x = torch.sin(self.world_theta) * world_translate_y + torch.cos(
            self.world_theta) * world_translate_x
        world_rotate_y = torch.cos(self.world_theta) * world_translate_y - torch.sin(
            self.world_theta) * world_translate_x
        world_rotate_z = world_translate_z
        camera_pred_x = 160 + f_x * (-world_rotate_y) / (world_rotate_x)
        camera_pred_y = 120 + f_y * -(world_rotate_z) / (world_rotate_x)  # using image coords y=0 means top
        return CameraPoint(camera_pred_x, camera_pred_y)

    def world_to_bounding_boxes(self, world_object):
        camera_pred_centre = self.world_to_camera(world_object.centre)
        camera_pred_top_left = self.world_to_camera(world_object.top_left)
        camera_pred_top_right = self.world_to_camera(world_object.top_right)
        camera_pred_bottom_left = self.world_to_camera(world_object.bottom_left)
        camera_pred_bottom_right = self.world_to_camera(world_object.bottom_right)

        pred_left = (camera_pred_bottom_left.x + camera_pred_top_left.x)/2
        pred_right = (camera_pred_bottom_right.x + camera_pred_top_right.x)/2
        pred_top = (camera_pred_top_left.y + camera_pred_top_right.y)/2
        pred_bottom = (camera_pred_bottom_left.y + camera_pred_bottom_right.y) / 2
        camera_pred_width = torch.clip(pred_right - pred_left, min=0.0)
        camera_pred_height = torch.clip(pred_bottom - pred_top, min=0.0)
        bounding_boxes = BoundingBoxes(camera_pred_centre.x, camera_pred_centre.y, camera_pred_width, camera_pred_height)
        return bounding_boxes

    def publish_annotated_image(self, filtered_detections, header, image):
        """Draws the bounding boxes on the image and publishes to /annotated_image"""

        if len(filtered_detections) > 0:
            pred_boxes = torch.zeros([len(filtered_detections.labels), 4])
            pred_boxes[:, 0] = filtered_detections.bounding_boxes.center_x - filtered_detections.bounding_boxes.width/2
            pred_boxes[:, 1] = filtered_detections.bounding_boxes.center_y - filtered_detections.bounding_boxes.height/2
            pred_boxes[:, 2] = filtered_detections.bounding_boxes.center_x + filtered_detections.bounding_boxes.width/2
            pred_boxes[:, 3] = filtered_detections.bounding_boxes.center_y + filtered_detections.bounding_boxes.height/2
            annotated_image = draw_bounding_boxes(torch.tensor(image), pred_boxes,
                                                  filtered_detections.labels, colors="yellow")
        else:
            annotated_image = torch.tensor(image)
        ros2_image_msg = self.bridge.cv2_to_imgmsg(annotated_image.numpy().transpose(1, 2, 0),
                                                   encoding="rgb8")
        ros2_image_msg.header = header
        self.annotated_image_publisher.publish(ros2_image_msg)
        return annotated_image

    def detect_image(self, image, header):
        detection_threshold = 0.9
        batch_image = np.expand_dims(image, axis=0)
        tensor_image = torch.tensor(batch_image/255.0, dtype=torch.float)
        mobilenet_detections = self.model(tensor_image)[0]  # pylint: disable=E1102 disable not callable warning

        filtered_idx = mobilenet_detections["scores"] > detection_threshold
        labels = mobilenet_detections["labels"][filtered_idx]
        boxes = mobilenet_detections["boxes"][filtered_idx]
        scores = mobilenet_detections["scores"][filtered_idx]
        pred_labels = [self.class_labels[labels[label_idx]] for label_idx in range(labels.shape[0])]

        center_x = (boxes[:, 0] + boxes[:, 2])/2.0
        center_y = (boxes[:,1] + boxes[:,3])/2.0
        width = boxes[:,2] - boxes[:,0]
        height = boxes[:, 3] - boxes[:, 1]
        filtered_detections = Detections(pred_labels, BoundingBoxes(center_x, center_y, width, height), scores)

        annotated_image = self.publish_annotated_image(filtered_detections, header, image)

        if filtered_detections is None:
            return torch.zeros([self.num_grid_cells, self.num_grid_cells, self.num_orientation_cells]) \
                    + \
                (1.0 / (self.num_grid_cells * self.num_grid_cells * self.num_orientation_cells))
        return filtered_detections, annotated_image

    def mobilenet_to_ros2(self, detection, header):
        """Converts a Detection tuple(label, bbox, score) to a ROS2 Detection2D message."""

        detection2d = Detection2D()
        detection2d.header = header
        object_hypothesis_with_pose = ObjectHypothesisWithPose()
        object_hypothesis = ObjectHypothesis()
        object_hypothesis.class_id = self.class_labels[detection.label]
        object_hypothesis.score = detection.score.detach().item()
        object_hypothesis_with_pose.hypothesis = object_hypothesis
        detection2d.results.append(object_hypothesis_with_pose)
        bounding_box = BoundingBox2D()
        bounding_box.center.position.x = float((detection.bbox[0] + detection.bbox[2]) / 2)
        bounding_box.center.position.y = float((detection.bbox[1] + detection.bbox[3]) / 2)
        bounding_box.center.theta = 0.0
        bounding_box.size_x = float(2 * (bounding_box.center.position.x - detection.bbox[0]))
        bounding_box.size_y = float(2 * (bounding_box.center.position.y - detection.bbox[1]))
        detection2d.bbox = bounding_box
        return detection2d
