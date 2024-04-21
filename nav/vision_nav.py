import collections
import itertools
import math
import torch

f_x = 494
f_y = 294
WorldPoint = collections.namedtuple("WorldPoint", "x, y, z")
WorldObject = collections.namedtuple("WorldObject", "name, centre, bottom_left, bottom_right, top_left, top_right")
CameraPoint = collections.namedtuple("CameraPoint", "x, y")
BoundingBoxes = collections.namedtuple("BoundingBoxes", "center, width, height")  # Grid of bounding boxes
ObjectDetection = collections.namedtuple("ObjectDetection", "bounding_boxes, detection_probabilities")


class VisionNav():
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
        self.world_z = .22

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
        if detections_msg is None:
            return torch.zeros([self.num_grid_cells, self.num_grid_cells, self.num_orientation_cells]) \
                    + \
                (1.0 / (self.num_grid_cells * self.num_grid_cells * self.num_orientation_cells))
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
