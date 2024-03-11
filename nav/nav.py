import collections
import math
import numpy as np
import scipy
import itertools

from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
import rclpy
from rclpy.node import Node
import torch
from scipy.spatial.transform import Rotation as R
from tf_transformations import euler_from_quaternion
from cv_bridge import CvBridge
from torchvision.utils import draw_bounding_boxes
import inertial_nav as inertial_nav_mod


Detection = collections.namedtuple("Detection", "label, bbox, score")


f_x = 494
f_y = 294
WorldPoint = collections.namedtuple("WorldPoint", "x, y, z")
WorldObject = collections.namedtuple("WorldObject", "name, centre, bottom_left, bottom_right, top_left, top_right")
CameraPoint = collections.namedtuple("CameraPoint", "x, y")
BoundingBoxes = collections.namedtuple("BoundingBoxes", "center, width, height")  # Grid of bounding boxes
ObjectDetection = collections.namedtuple("ObjectDetection", "bounding_boxes, detection_probabilities")


class PoseNav():
    def __init__(self, num_grid_cells, num_orientation_cells):
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
        self.world_z = .24

        self.world_position = (torch.arange(self.num_grid_cells)+0.5) * self.world_cell_size - self.world_grid_length/2.0
        self.world_xs = self.world_position
        self.world_ys = torch.flip(self.world_position, dims=[0])
        self.world_thetas = torch.arange(self.num_orientation_cells)*2*math.pi/self.num_orientation_cells
        self.world_x, self.world_y, self.world_theta = torch.meshgrid(self.world_xs, self.world_ys, self.world_thetas, indexing='xy')

        self.world_dog_boxes = self.world_to_bbox(self.world_dog)
        self.world_cat_boxes = self.world_to_bbox(self.world_cat)
        self.dog = ObjectDetection(self.world_to_bbox(self.world_dog), self.box_probability(self.world_dog_boxes))
        self.cat = ObjectDetection( self.world_to_bbox(self.world_cat), self.box_probability(self.world_cat_boxes))
        self.object_dictionary = {"dog": ObjectDetection(self.world_to_bbox(self.world_dog), self.box_probability(self.world_dog_boxes)),
            "cat": ObjectDetection(self.world_to_bbox(self.world_cat), self.box_probability(self.world_cat_boxes)) }
        self.object_list = list(self.object_dictionary.keys())

    def prob_map(self, bbox, boxes):
        scale = 25.0
        res1 = torch.distributions.normal.Normal(boxes.center.x, scale).log_prob(torch.tensor(bbox.center.position.x))
        res2 = torch.distributions.normal.Normal(boxes.width, scale).log_prob(torch.tensor(bbox.size_x))
        res3 = torch.distributions.normal.Normal(boxes.height, scale).log_prob(torch.tensor(bbox.size_y))
        res4 = torch.distributions.normal.Normal(boxes.center.y, scale).log_prob(
            torch.tensor(bbox.center.position.y))
        res = res1 + res2 + res3 + res4
        mynorm = res - torch.logsumexp(res, dim=[0, 1, 2], keepdim=False)
        smyprobs = torch.exp(mynorm)
        return smyprobs

    def probmessage_cond_a(self, detections_msg, proposals):
        # detections is Detections list, proposals is list of world_boxes
        # loop through proposals assign to a detection and recurse
        prob_dist_random = 0.05 * 0.01 * 0.01 * 0.01 * 0.01
        prob_dist_random_boxes = prob_dist_random+(self.world_x*0.0)
        if proposals == []:
            return prob_dist_random_boxes**len(detections_msg)
        proposal, *remaining_proposals = proposals
        cum_prob = self.world_x * 0.0
        for assign_idx in range(len(detections_msg)):
            rem_detections = detections_msg[:assign_idx] + detections_msg[assign_idx+1:]
            proposal_name = self.object_list[proposal]
            prob_assignment = self.prob_map(detections_msg[assign_idx].bbox, self.object_dictionary[proposal_name].bounding_boxes)
            if proposal_name != detections_msg[assign_idx].results[0].hypothesis.class_id:
                prob_assignment = prob_assignment * 0.0
            rem_prob = self.probmessage_cond_a(rem_detections, remaining_proposals)
            total_prob = prob_assignment * rem_prob
            cum_prob += total_prob / len(detections_msg)
        return cum_prob

    def probmessage(self, detections_msg):
        comb = list(itertools.product([False,True], repeat=2))
        s = self.world_x * 0.0
        for assignment in comb:
            probs = self.world_x * 0.0 + 1.0
            for idx in range(len(assignment)):
                if idx == False:
                    probs = probs * (1.0-self.object_dictionary[self.object_list[idx]].detection_probabilities)
                else:
                    probs = probs * self.object_dictionary[self.object_list[idx]].detection_probabilities
            assignments = [i for i, x in enumerate(assignment) if x]
            s += self.probmessage_cond_a(detections_msg, assignments) * probs
        return s

    def box_probability(self, boxes):
        """Computes the probability of a bounding box being detected.
        Example: If the bounding box is completely outside the camera field of view, it won't be detected.
        """
        cons_camera_left_x = torch.clip(boxes.center.x - boxes.width, min=-160, max=+160)
        cons_camera_right_x = torch.clip(boxes.center.x + boxes.width, min=-160, max=+160)
        cons_camera_bottom_y = torch.clip(boxes.center.y - boxes.height, min=-120, max=+120)
        cons_camera_top_y = torch.clip(boxes.center.y + boxes.height, min=-120, max=+120)
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

    def world_to_bbox(self, world_object):
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
        bounding_boxes = BoundingBoxes(camera_pred_centre, camera_pred_width, camera_pred_height)
        return bounding_boxes


class Nav(Node):
    def __init__(self):
        super().__init__("nav")
        self.detections_subscription = self.create_subscription(
            Detection2DArray,
            "/detected_objects",
            self.detections_callback,
            10)
        self.annotated_image_subscription = self.create_subscription(
            Image,
            "/annotated_image",
            self.annotated_image_callback,
            10)
        self.pose_subscription = self.create_subscription(
            Odometry,
            "/differential_drive_controller/odom",
            self.pose_callback,
            10)
        self.probmap_publisher = \
            self.create_publisher(OccupancyGrid, "probmap", 10)
        self.pose_publisher = \
            self.create_publisher(PoseStamped, "nav_pose", 10)
        self.debug_image_publisher = \
            self.create_publisher(Image, "debug_image", 10)

        self.num_grid_cells = 101
        self.num_orientation_cells = 128
        self.grid_cells_origin_x = -1.5
        self.grid_cells_origin_y = -1.5
        self.world_grid_length = 3.0
        self.world_cell_size = self.world_grid_length/self.num_grid_cells
        self.last_inertial_position = None
        self.state = 0
        self.detections = None
        self.bridge = CvBridge()
        self.inertial_nav = inertial_nav_mod.InertialNav(self.num_grid_cells, self.num_orientation_cells, "Uniform")
        self.pose_nav = PoseNav(self.num_grid_cells, self.num_orientation_cells)



    def publish_occupancy_grid_msg(self, pose_probability_map, header):
        myprobs = torch.sum(pose_probability_map, axis=2)
        kernel = torch.ones([1, 1, 3, 3])
        conv = torch.nn.functional.conv2d(myprobs.unsqueeze(0), kernel, padding=1)[0]
        prob_map_msg = OccupancyGrid()
        prob_map_msg.header = header
        prob_map_msg.header.frame_id = "base_link"
        prob_map_msg.info.resolution = self.world_cell_size
        prob_map_msg.info.width = self.num_grid_cells
        prob_map_msg.info.height = self.num_grid_cells
        prob_map_msg.info.origin.position.x = self.grid_cells_origin_x
        prob_map_msg.info.origin.position.y = self.grid_cells_origin_y
        conv = conv/conv.max()
        prob_map_msg.data = torch.flip(100.0 * conv, dims=[0]).type(torch.int).flatten().tolist()
        self.probmap_publisher.publish(prob_map_msg)

    def get_location_MLE(self, pose_probability_map):
        loc = (pose_probability_map==torch.max(pose_probability_map)).nonzero()[0]
        orientation = pose_probability_map[loc[0], loc[1]].argmax()
        return (loc, orientation)

    def publish_pose_msg(self, pose_probability_map, header):
        (loc, orientation) = self.get_location_MLE(pose_probability_map)
        pose_msg = Pose()
        point = Point()
        point.x = float(self.grid_cells_origin_x + loc[1]*self.world_cell_size)
        point.y = float(self.grid_cells_origin_y + self.world_grid_length - (loc[0]*self.world_cell_size))
        pose_msg.position = point
        quaternion = R.from_euler('xyz', [0, 0, (orientation/self.num_orientation_cells)*2*3.141]).as_quat()
        q = Quaternion()
        q.x, q.y, q.z, q.w = quaternion
        pose_msg.orientation = q
        pose_stamped = PoseStamped()
        pose_stamped.header = header
        pose_stamped.pose = pose_msg
        self.pose_publisher.publish(pose_stamped)

    def annotated_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        image = cv_image.copy().transpose((2, 0, 1))
        self.annotated_image = image

    def publish_vision_debug_image(self, pose_probability_map, header):
        return
        (loc, orientation) = self.get_location_MLE(pose_probability_map)
        box_tensor = torch.tensor([self.dog_boxes.center.x[loc[0], loc[1], orientation]])
        score = self.dog_boxes_probability[loc[0], loc[1], orientation]
        box = Detection("debug", box_tensor, score)
        center_x = self.dog_boxes.center.x[loc[0], loc[1], orientation]
        center_y = self.dog_boxes.center.y[loc[0], loc[1], orientation]
        width = self.dog_boxes.width[loc[0], loc[1], orientation]
        height = self.dog_boxes.height[loc[0], loc[1], orientation]
        xmin = center_x - width / 2
        xmax = center_x + width / 2
        ymin = center_y - height / 2
        ymax = center_y + height / 2
        box = torch.tensor([[xmin, ymin, xmax, ymax]])
        print("Box=", box)
        debug_image = draw_bounding_boxes(torch.tensor(self.annotated_image), box,
                                              ["debug"], colors="green")
        ros2_image_msg = self.bridge.cv2_to_imgmsg(debug_image.numpy().transpose(1, 2, 0), encoding = "rgb8")
        ros2_image_msg.header = header
        self.debug_image_publisher.publish(ros2_image_msg)

    def detections_callback(self, msg):
        self.detections = msg.detections

    def pose_callback(self, msg):
        if self.detections is None:
            return
        print("POSE", msg.pose.pose.position)
        current_inertial_position = torch.tensor([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        current_inertial_orientation = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        pose_from_detections_probability_map = self.pose_nav.probmessage(self.detections)
        if self.last_inertial_position is None:
            self.last_inertial_position = current_inertial_position
            self.last_inertial_orientation = current_inertial_orientation
            self.inertial_nav.current_probability_map = pose_from_detections_probability_map
            return
        moving = self.inertial_nav.inertial_update(self.last_inertial_position, current_inertial_position, self.last_inertial_orientation, current_inertial_orientation)
        self.last_inertial_position = current_inertial_position
        self.last_inertial_orientation = current_inertial_orientation
        if moving:
            print("Moving")
            self.state = 0
        else:
            if self.state == 0:
                self.publish_vision_debug_image(pose_from_detections_probability_map, msg.header)
                self.inertial_nav.update_from_sensor(pose_from_detections_probability_map)
                self.state = 1
        header = msg.header
        self.publish_occupancy_grid_msg(self.inertial_nav.current_probability_map, header)
        self.publish_pose_msg(self.inertial_nav.current_probability_map, header)


def test1(node):
    bounding_box = BoundingBox2D()
    bounding_box.center.position.x = 160.0
    bounding_box.center.position.y = 120.0
    bounding_box.size_x = 84.0
    bounding_box.size_y = 104.0
    node.prob_map(bounding_box)

def test2(node):
    bounding_box = BoundingBox2D()
    bounding_box.center.position.x = 181.0
    bounding_box.center.position.y = 121.0
    bounding_box.size_x = 122.0
    bounding_box.size_y = 227.0
#    node.probmessage(msg.detections)


rclpy.init()
nav_node = Nav()
test2(nav_node)
rclpy.spin(nav_node)
nav_node.destroy_node()
rclpy.shutdown()
