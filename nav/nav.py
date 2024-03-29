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
import vision_nav as vision_nav_mod

Detection = collections.namedtuple("Detection", "label, bbox, score")






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
        self.vision_nav = vision_nav_mod.VisionNav(self.num_grid_cells, self.num_orientation_cells)



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
        pose_from_detections_probability_map = self.vision_nav.probmessage(self.detections)
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
