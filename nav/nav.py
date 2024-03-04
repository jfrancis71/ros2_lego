import collections
import math
import numpy as np
import scipy

from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import BoundingBox2D
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
import rclpy
from rclpy.node import Node
import torch
from scipy.spatial.transform import Rotation as R
from tf_transformations import euler_from_quaternion



f_x = 494
f_y = 294
WorldPoint = collections.namedtuple("WorldPoint", "x, y, z")
WorldObject = collections.namedtuple("WorldObject", "name, centre, bottom_left, bottom_right, top_left, top_right")
CameraPoint = collections.namedtuple("CameraPoint", "x, y")
BoundingBoxes = collections.namedtuple("BoundingBoxes", "center, width, height")  # Grid of bounding boxes


class Nav(Node):
    def __init__(self):
        super().__init__("nav")
        self.image_subscription = self.create_subscription(
            Detection2DArray,
            "/detected_objects",
            self.detections_callback,
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

        self.world_dog = WorldObject("dog",
            WorldPoint(1.5, 0.0, 0.27),
            WorldPoint(1.5, 0.11, .02),
            WorldPoint(1.5, -0.11, .02),
            WorldPoint(1.5, 0.11, .52),
            WorldPoint(1.5, -0.11, .52)
        )
        self.num_grid_cells = 101
        self.num_orientation_cells = 128
        self.world_grid_length = 3.0
        self.world_cell_size = self.world_grid_length/self.num_grid_cells
        self.grid_cells_origin_x = -1.5
        self.grid_cells_origin_y = -1.5

        self.world_z = .24

        self.world_position = (torch.arange(self.num_grid_cells)+0.5) * self.world_cell_size - self.world_grid_length/2.0
        self.world_xs = self.world_position
        self.world_ys = torch.flip(self.world_position, dims=[0])
        self.world_thetas = torch.arange(self.num_orientation_cells)*2*math.pi/self.num_orientation_cells
        self.world_x, self.world_y, self.world_theta = torch.meshgrid(self.world_xs, self.world_ys, self.world_thetas, indexing='xy')

        self.boxes = self.world_to_bbox(self.world_dog)
        cons_camera_left_x = torch.clip(self.boxes.center.x - self.boxes.width, min=-160, max=+160)
        cons_camera_right_x = torch.clip(self.boxes.center.x + self.boxes.width, min=-160, max=+160)
        cons_camera_bottom_y = torch.clip(self.boxes.center.y - self.boxes.height, min=-120, max=+120)
        cons_camera_top_y = torch.clip(self.boxes.center.y + self.boxes.height, min=-120, max=+120)
        cons_area = (cons_camera_right_x-cons_camera_left_x)*(cons_camera_top_y-cons_camera_bottom_y)
        area_ratio = cons_area / ((self.boxes.width*self.boxes.height)+cons_area)
        area_ratio = torch.nan_to_num(area_ratio, nan=0.0)
        self.proba = 0.05 * (1-area_ratio) + 0.95 * area_ratio
        self.current_probability_map = \
            torch.zeros([self.num_grid_cells, self.num_grid_cells, self.num_orientation_cells])
#             + \
#            (1.0/(self.num_grid_cells*self.num_grid_cells*self.num_orientation_cells)))
        self.current_probability_map[int(self.num_grid_cells/2), int(self.num_grid_cells/2), 0] = 1.0
        self.last_inertial_position = None
        self.position_kernel = np.zeros([11,11])
        self.orientation_kernel = np.zeros([self.num_orientation_cells])
        self.position_kernel[5,5] = 1.0
        self.orientation_kernel[0] = 1.0

    def world_to_camera(self, world_point):
        world_translate_x = world_point.x - self.world_x
        world_translate_y = world_point.y - self.world_y
        world_translate_z = world_point.z - self.world_z
        world_rotate_x = torch.sin(self.world_theta) * world_translate_y + torch.cos(
            self.world_theta) * world_translate_x
        world_rotate_y = torch.cos(self.world_theta) * world_translate_y - torch.sin(
            self.world_theta) * world_translate_x
        world_rotate_z = world_translate_z
        camera_pred_x = f_x * (-world_rotate_y) / (world_rotate_x)
        camera_pred_y = f_y * -(world_rotate_z) / (world_rotate_x)  # using image coords y=0 means top
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

    def prob_map(self, world_object, bbox):
        scale = 25.0
        res1 = torch.distributions.normal.Normal(self.boxes.center.x, scale).log_prob(torch.tensor(bbox.center.position.x-160))
        res2 = torch.distributions.normal.Normal(self.boxes.width, scale).log_prob(torch.tensor(bbox.size_x))
        res3 = torch.distributions.normal.Normal(self.boxes.height, scale).log_prob(torch.tensor(bbox.size_y))
        res4 = torch.distributions.normal.Normal(self.boxes.center.y, scale).log_prob(
            torch.tensor(bbox.center.position.y - 120))
        res = res1 + res2 + res3 + res4
        mynorm = res - torch.logsumexp(res, dim=[0, 1, 2], keepdim=False)
        smyprobs = torch.exp(mynorm)
        return smyprobs

    def probmessage_cond_a(self, detections, assignment):
        prob_dist_random = 0.05 * 0.01 * 0.01 * 0.01 * 0.01
        prob_dist_random_boxes = prob_dist_random+(self.world_x*0.0)
        if assignment == 0:
            probs = prob_dist_random_boxes**len(detections)
        else:
            probs = self.world_x*0.0
            for m in detections:
                if m.results[0].hypothesis.class_id == "dog":
                    probs += prob_dist_random_boxes**(len(detections)-1) * self.prob_map(self.world_dog, m.bbox)
        return probs

    def probmessage(self, detections):
        probs1 = self.world_x * 0.0
        probs1 += self.probmessage_cond_a(detections, 0)
        if len(detections) >= 1:
            probs1 += self.proba * self.probmessage_cond_a(detections, 1)
        norm_probs = probs1 / probs1.sum()
        return norm_probs

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
        prob_map_msg.data = torch.flip(100.0 * conv, dims=[0]).type(torch.int).flatten().tolist()
        self.probmap_publisher.publish(prob_map_msg)

    def publish_pose_msg(self, pose_probability_map, header):
        loc = (pose_probability_map==torch.max(pose_probability_map)).nonzero()[0]
        orientation = pose_probability_map[loc[0], loc[1]].argmax()
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

    def detections_callback(self, msg):
#        self.pose_from_detections_probability_map = self.probmessage(msg.detections)
#        self.publish_occupancy_grid_msg(self.pose_from_detections_probability_map, msg.header)
#        self.publish_pose_msg(self.pose_from_detections_probability_map, msg.header)
#        print("loc")
        pass

    def pose_callback(self, msg):
        print("POSE", msg.pose.pose.position)
        current_inertial_position = torch.tensor([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation

        current_inertial_orientation = -euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        print("o=", current_inertial_orientation)
        if self.last_inertial_position is None:
            self.last_inertial_position = current_inertial_position
            self.last_inertial_orientation = current_inertial_orientation
            return
        inertial_position_difference = self.last_inertial_position - current_inertial_position
        inertial_orientation_difference = self.last_inertial_orientation - current_inertial_orientation
        tensor_orientation = torch.tensor(current_inertial_orientation)
        inertial_forward = torch.inner(torch.stack([tensor_orientation.cos(), tensor_orientation.sin()]), inertial_position_difference)
        self.last_inertial_position = current_inertial_position
        self.last_inertial_orientation = current_inertial_orientation
        for r in range(self.num_orientation_cells):
            shifted = scipy.ndimage.shift(self.position_kernel, (0.0, inertial_forward/self.world_cell_size), output=None, order=3, mode='constant', cval=0.0, prefilter=True)
            rotated = scipy.ndimage.rotate(shifted, -360 * r/self.num_orientation_cells, reshape=False)
            self.current_probability_map[:,:,r] = torch.nn.functional.conv2d(self.current_probability_map[:,:,r:r+1].permute([2,0,1]), torch.tensor(rotated, dtype=torch.float).reshape([1,1,11,11]), padding="same").permute([1,2,0])[:,:,0]
        #    shifted_orientation = scipy.ndimage.shift(self.orientation_kernel, inertial_orientation_difference * self.num_orientation_cells/ (2*3.141),
        #                                 output=None, order=3, mode='wrap', cval=0.0, prefilter=True)

        self.current_probability_map = torch.tensor(
            scipy.ndimage.shift(self.current_probability_map, (0,0, inertial_orientation_difference * self.num_orientation_cells/ (2*3.141)), mode='wrap'),
            dtype=torch.float)

        #self.current_probability_map = torch.nn.functional.conv2d(
        #    self.current_probability_map.permute([2,0,1]).reshape([1,self.num_orientation_cells, self.num_grid_cells, self.num_grid_cells]),
        #    torch.tensor(shifted_orientation, dtype=torch.float).reshape([1, self.num_orientation_cells, 1, 1]))
        # rescale
        self.current_probability_map = torch.clip(self.current_probability_map, min=0.0)
        self.current_probability_map = self.current_probability_map / self.current_probability_map.sum()
        if inertial_position_difference.norm() > .001 or abs(inertial_orientation_difference) > .001:
            print("Moving")
        header = msg.header
        self.publish_occupancy_grid_msg(self.current_probability_map, header)
        self.publish_pose_msg(self.current_probability_map, header)


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
