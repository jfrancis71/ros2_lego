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
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


Detection = collections.namedtuple("Detection", "label, bbox, score")

torch.set_num_threads(1)




class Nav(Node):
    def __init__(self):
        super().__init__("nav")
        self.stationary, self.stopped, self.moving = range(3)
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
            self.odometry_callback,
            10)
        self.probmap_publisher = \
            self.create_publisher(OccupancyGrid, "probmap", 10)
        self.pose_publisher = \
            self.create_publisher(PoseStamped, "nav_pose", 10)
        self.debug_image_publisher = \
            self.create_publisher(Image, "debug_image", 10)

        self.num_grid_cells = 101
        self.num_orientation_cells = 64
        self.grid_cells_origin_x = -1.5
        self.grid_cells_origin_y = -1.5
        self.world_grid_length = 3.0
        self.world_cell_size = self.world_grid_length/self.num_grid_cells
        self.last_inertial_position = None
        self.annotated_image = None
        self.state = self.stopped
        self.detections = None
        self.bridge = CvBridge()
        self.inertial_nav = inertial_nav_mod.InertialNav(self.num_grid_cells, self.num_orientation_cells, "Uniform")
        self.vision_nav = vision_nav_mod.VisionNav(self.num_grid_cells, self.num_orientation_cells)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_occupancy_grid_msg(self.inertial_nav.current_probability_map, PoseStamped().header)


    def publish_occupancy_grid_msg(self, pose_probability_map, header):
        myprobs = torch.sum(pose_probability_map, axis=2)
        kernel = torch.ones([1, 1, 3, 3])
        conv = torch.nn.functional.conv2d(myprobs.unsqueeze(0), kernel, padding=1)[0]
        prob_map_msg = OccupancyGrid()
        prob_map_msg.header = header
        prob_map_msg.header.frame_id = "map"
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

    def publish_pose_msg1(self, pose_probability_map, header):
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
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose = pose_msg
        self.pose_publisher.publish(pose_stamped)

    def publish_pose_msg(self, pose_probability_map, header):
        (loc, orientation) = self.get_location_MLE(pose_probability_map)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'magnetic_north'
        t.header = header
        t.header.frame_id = "map"
        t.child_frame_id = 'base_footprint'
        quaternion = R.from_euler('xyz', [0, 0, (orientation/self.num_orientation_cells)*2*3.141]).as_quat()
        q = Quaternion()
        q.x, q.y, q.z, q.w = quaternion
        t.transform.rotation = q
        t.transform.translation.x = float(self.grid_cells_origin_x + loc[1]*self.world_cell_size)
        t.transform.translation.y = float(self.grid_cells_origin_y + self.world_grid_length - (loc[0]*self.world_cell_size))
        self.tf_static_broadcaster.sendTransform(t)
        self.publish_pose_msg1(pose_probability_map, header)

    def annotated_image_callback(self, msg):
        print("Other callback")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        image = cv_image.copy().transpose((2, 0, 1))
        self.annotated_image = image

    def publish_vision_debug_image(self, pose_probability_map, detections, header):
#        return
        (loc, orientation) = self.get_location_MLE(pose_probability_map)
        boxes = []
        labels = []
        for obj_label in self.vision_nav.object_list:
            obj = self.vision_nav.object_dictionary[obj_label]
            score = obj.detection_probabilities[loc[0], loc[1], orientation]
            if score < .5:
                continue
#            box = Detection("debug", box_tensor, score)
            center_x = obj.bounding_boxes.center.x[loc[0], loc[1], orientation]
            center_y = obj.bounding_boxes.center.y[loc[0], loc[1], orientation]
            width = obj.bounding_boxes.width[loc[0], loc[1], orientation]
            height = obj.bounding_boxes.height[loc[0], loc[1], orientation]
            xmin = center_x - width / 2
            xmax = center_x + width / 2
            ymin = center_y - height / 2
            ymax = center_y + height / 2
            box = torch.stack([xmin, ymin, xmax, ymax])
            labels.append("debug_"+obj_label)
            boxes.append(box)
        if len(boxes) != 0:
            tensor_boxes = torch.stack(boxes)
            print("Box=", boxes)
            debug_image = draw_bounding_boxes(torch.tensor(self.annotated_image), tensor_boxes,
                                              labels, colors="green")
        else:
            if self.annotated_image is None:
                return
            debug_image = torch.tensor(self.annotated_image)

        ros2_image_msg = self.bridge.cv2_to_imgmsg(debug_image.numpy().transpose(1, 2, 0), encoding = "rgb8")
        ros2_image_msg.header = header
        self.debug_image_publisher.publish(ros2_image_msg)

    def detections_callback(self, msg):
        print("callback")
        if self.state == self.stopped:
            pose_from_detections_probability_map = self.vision_nav.probmessage(msg.detections)
            self.inertial_nav.update_from_sensor(pose_from_detections_probability_map)
            if self.annotated_image is not None:
                self.publish_vision_debug_image(pose_from_detections_probability_map, msg.detections, msg.header)
            self.state = self.stationary

    def odometry_callback(self, msg):
#        if self.detections is None:
#            return
        print("POSE", msg.pose.pose.position)
        current_inertial_position = torch.tensor([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        current_inertial_orientation = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        print("Current IO=", current_inertial_orientation)

        if self.last_inertial_position is None:
            self.last_inertial_position = current_inertial_position
            self.last_inertial_orientation = current_inertial_orientation
            return
        moving = self.inertial_nav.inertial_update(self.last_inertial_position, current_inertial_position, self.last_inertial_orientation, current_inertial_orientation)
        self.last_inertial_position = current_inertial_position
        self.last_inertial_orientation = current_inertial_orientation
        if moving:
            print("Moving")
            self.state = self.moving
        else:
            if self.state == self.moving:
                self.state = self.stopped
        header = msg.header

        pmap = self.inertial_nav.getpmap()
        self.publish_occupancy_grid_msg(pmap, header)
        self.publish_pose_msg(pmap, header)


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
