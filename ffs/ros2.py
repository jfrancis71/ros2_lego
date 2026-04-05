import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import CameraInfo

import struct

from Utils import (
    AMP_DTYPE, set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)

from core.utils.utils import InputPadder


class FFSNode(Node):
    def __init__(self):
        super().__init__("ffs_node")
        self.left_image_sub = Subscriber(self, Image, "/left/image_rect_color")
        self.right_image_sub = Subscriber(self, Image, "/right/image_rect_color")
        self.left_image_info_sub = Subscriber(self, CameraInfo, "/left/camera_info")
        self.right_image_info_sub = Subscriber(self, CameraInfo, "/right/camera_info")
        self.image_publisher = \
                self.create_publisher(Image, "depth_image", 1)
        self.pcd_publisher = self.create_publisher(PointCloud2, 'ffs_pcd', 1)
        self.bridge = CvBridge()
        self.model = torch.load("/root/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth", map_location='cpu', weights_only=False)
        self.model.cuda().eval()
        queue_size = 10
        self.sync = TimeSynchronizer([self.left_image_sub, self.right_image_sub, self.left_image_info_sub, self.right_image_info_sub], queue_size)
        self.sync.registerCallback(self.SyncCallback)
        self.get_logger().info("Node has started.")

    def convert(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        image = cv_image.copy().transpose((2, 0, 1))
        return image

    def SyncCallback(self, left_msg, right_msg, left_info_msg, right_info_msg):
        left_image = self.convert(left_msg)
        right_image = self.convert(right_msg)
        left_batch_image = np.expand_dims(left_image, axis=0)
        left_tensor_image = torch.tensor(left_batch_image, dtype=torch.float32, device="cuda")
        right_batch_image = np.expand_dims(right_image, axis=0)
        right_tensor_image = torch.tensor(right_batch_image, dtype=torch.float32, device="cuda")

        padder = InputPadder(left_tensor_image.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(left_tensor_image, right_tensor_image)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = self.model.forward(img0, img1, iters=4, test_mode=True, optimize_build_volume='pytorch1')
            tens = torch.tensor(disp, dtype=torch.uint8)
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(left_info_msg.height,left_info_msg.width).clip(0, None)

        cmap = None
        min_val = None
        max_val = None
        vis = vis_disparity(disp, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)
        left_orig = left_image.copy().transpose((1, 2, 0))
        right_orig = right_image.copy().transpose((1, 2, 0))
        vis = np.concatenate([left_orig, right_orig, vis], axis=1)
        s = 1280/vis.shape[1]
        resized_vis = cv2.resize(vis, (int(vis.shape[1]*s), int(vis.shape[0]*s)))

        ros2_image_msg = self.bridge.cv2_to_imgmsg(resized_vis,
                                                   encoding="rgb8")
        ros2_image_msg.header = left_msg.header
        self.image_publisher.publish(ros2_image_msg)

        K = left_info_msg.k.reshape(3,3)
        baseline = -right_info_msg.p[3] / right_info_msg.p[0]
        depth = K[0,0]*baseline/disp
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), left_orig.reshape(-1,3))
        ros_dtype = PointField.FLOAT32

        points = np.array(pcd.points)
        colors = np.array(pcd.colors)

        itemsize = np.dtype(np.float32).itemsize

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        point_struct = struct.Struct("<fffBBB")
        num_points = points.shape[0]
        buffer = bytearray(point_struct.size * num_points)
        for i in range(num_points):
            r, g, b = colors[i]
            r, g, b = colors[i, 0]*255, colors[i, 1]*255, colors[i, 2]*255
            r, g, b = int(r), int(g), int(b)
            point_struct.pack_into(
                buffer, i * point_struct.size, points[i,0], points[i,1], points[i,2], b, g, r
        )

        pcd_msg = PointCloud2(
            header=left_msg.header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=15,
            row_step=(itemsize * 3 * points.shape[0]),
            data=buffer
        )
        self.pcd_publisher.publish(pcd_msg)


torch.autograd.set_grad_enabled(False)

rclpy.init()
ffs_node = FFSNode()
rclpy.spin(ffs_node)
ffs_node.destroy_node()
rclpy.shutdown()
