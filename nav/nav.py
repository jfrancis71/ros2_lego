import collections
import math

from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import BoundingBox2D
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
import torch


f = 428
WorldPoint = collections.namedtuple("WorldPoint", "x, y, z")
WorldObject = collections.namedtuple("WorldObject", "centre, bottom_left, bottom_right, top_left, top_right")
CameraPoint = collections.namedtuple("CameraPoint", "x, y")

class Nav(Node):
    def __init__(self):
        super().__init__("nav")
        self.subscription = self.create_subscription(
            Detection2DArray,
            "/detected_objects",
            self.listener_callback,
            10)
        self.probmap_publisher = \
            self.create_publisher(OccupancyGrid, "probmap", 10)
        self.world_object = WorldObject(
            WorldPoint(1.5, 0.0, 0.27),
            WorldPoint(1.5, 0.14, .02),
            WorldPoint(1.5, -0.14, .02),
            WorldPoint(1.5, 0.14, .52),
            WorldPoint(1.5, -0.14, .52)
        )
        self.num_grid_cells = 101
        self.num_orientation_cells = 16
        self.world_grid_length = 3.0
        self.world_cell_size = self.world_grid_length/self.num_grid_cells

        self.world_z = .24

        self.world_position = (torch.arange(self.num_grid_cells)+0.5) * self.world_cell_size - self.world_grid_length/2.0
        self.world_xs = self.world_position
        self.world_ys = torch.flip(self.world_position, dims=[0])
        self.world_thetas = torch.arange(self.num_orientation_cells)*2*math.pi/self.num_orientation_cells
        self.world_x, self.world_y, self.world_theta = torch.meshgrid(self.world_xs, self.world_ys, self.world_thetas, indexing='xy')


    def world_to_camera(self, world_point):
        world_translate_x = world_point.x - self.world_x
        world_translate_y = world_point.y - self.world_y
        world_translate_z = world_point.z - self.world_z
        world_rotate_x = torch.sin(self.world_theta) * world_translate_y + torch.cos(
            self.world_theta) * world_translate_x
        world_rotate_y = torch.cos(self.world_theta) * world_translate_y - torch.sin(
            self.world_theta) * world_translate_x
        world_rotate_z = world_translate_z
        camera_pred_x = f * (-world_rotate_y) / (world_rotate_x)
        camera_pred_y = f * (world_rotate_z) / (world_rotate_x)
        return CameraPoint(camera_pred_x, camera_pred_y)

    def prob_map(self, world_object, bbox):
        camera_pred_centre = self.world_to_camera(world_object.centre)
        camera_pred_top_left = self.world_to_camera(world_object.top_left)
        camera_pred_top_right = self.world_to_camera(world_object.top_right)
        camera_pred_bottom_left = self.world_to_camera(world_object.bottom_left)
        camera_pred_bottom_right = self.world_to_camera(world_object.bottom_right)

        pred_left = (camera_pred_bottom_left.x + camera_pred_top_left.x)/2
        pred_right = (camera_pred_bottom_right.x + camera_pred_top_right.x)/2
        pred_top = (camera_pred_top_left.y + camera_pred_top_right.y)/2
        pred_bottom = (camera_pred_bottom_left.y + camera_pred_bottom_right.y) / 2
        camera_pred_width = pred_right - pred_left
        camera_pred_height = pred_top - pred_bottom
        res1 = torch.distributions.normal.Normal(camera_pred_centre.x, 5).log_prob(torch.tensor(bbox.center.position.x))
        res2 = torch.distributions.normal.Normal(camera_pred_width, 5).log_prob(torch.tensor(bbox.size_x))
        res3 = torch.distributions.normal.Normal(camera_pred_height, 5).log_prob(torch.tensor(bbox.size_y))
        res = res1 + res2 + res3
        mynorm = res - torch.logsumexp(res, dim=[0, 1, 2], keepdim=False)
        smyprobs = torch.exp(mynorm)
        myprobs = torch.sum(smyprobs, axis=2)

        return myprobs

    def listener_callback(self, msg):
        for det in msg.detections:
            print(det)
            myprobs = self.prob_map(self.world_object, det.bbox)
            kernel = torch.ones([1, 1, 3, 3])
            conv = torch.nn.functional.conv2d(myprobs.unsqueeze(0), kernel, padding=1)[0]
            prob_map_msg = OccupancyGrid()
            prob_map_msg.header = msg.header
            prob_map_msg.header.frame_id = "base_link"
            prob_map_msg.info.resolution = 3/100
            prob_map_msg.info.width = 101
            prob_map_msg.info.height = 101
            prob_map_msg.info.origin.position.x = -1.5
            prob_map_msg.info.origin.position.y = -1.5
            prob_map_msg.data = (100 * torch.flip(conv, dims=[0])).type(torch.int).flatten().tolist()
            self.probmap_publisher.publish(prob_map_msg)

def test1(node):
    bounding_box = BoundingBox2D()
    bounding_box.center.position.x = 160.0
    bounding_box.center.position.y = 120.0
    bounding_box.size_x = 84.0
    bounding_box.size_y = 104.0
    node.prob_map(bounding_box)

def test2(node):
    bounding_box = BoundingBox2D()
    bounding_box.center.position.x = 176.0
    bounding_box.center.position.y = 96.0
    bounding_box.size_x = 76.0
    bounding_box.size_y = 112.0
    node.prob_map(node.world_object, bounding_box)


rclpy.init()
nav_node = Nav()
test2(nav_node)
rclpy.spin(nav_node)
nav_node.destroy_node()
rclpy.shutdown()
