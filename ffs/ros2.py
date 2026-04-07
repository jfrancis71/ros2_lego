import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from message_filters import Subscriber, TimeSynchronizer
from cv_bridge import CvBridge
import torch
from Utils import (
    AMP_DTYPE, vis_disparity,
    depth2xyzmap, toOpen3dCloud,
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
                self.create_publisher(Image, "/ffs_disparity", 1)
        self.pcd_publisher = self.create_publisher(PointCloud2, '/ffs_points2', 1)
        queue_size = 1
        self.sync = TimeSynchronizer([self.left_image_sub, self.right_image_sub, self.left_image_info_sub, self.right_image_info_sub], queue_size)
        self.sync.registerCallback(self.sync_cb)
        self.bridge = CvBridge()
        self.model = torch.load("/root/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth", map_location='cpu', weights_only=False)
        self.model.cuda().eval()
        self.get_logger().info("Node has started.")

    def move_image_to_gpu(self, image):
        image = image.transpose((2, 0, 1))
        batch_image = np.expand_dims(image, axis=0)
        return torch.tensor(batch_image, dtype=torch.float32, device="cuda")

    def publish_disparity_image(self, disparity_map, header):
        vis = vis_disparity(disparity_map, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
        ros2_image_msg = self.bridge.cv2_to_imgmsg(vis,
                                                   encoding="rgb8")
        ros2_image_msg.header = header
        self.image_publisher.publish(ros2_image_msg)

    def publish_point_cloud(self, disparity_map, left_img,
            left_info_msg, right_info_msg, header):
        K = left_info_msg.k.reshape(3,3)
        baseline = -right_info_msg.p[3] / right_info_msg.p[0]
        depth = K[0,0]*baseline/disparity_map
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), left_img.reshape(-1,3))
        ros_dtype = PointField.FLOAT32
        points = np.array(pcd.points, dtype=np.float32)
        colors = (np.array(pcd.colors, dtype=np.float32)*255).astype(dtype=np.uint8)
        colors = np.flip(colors, axis=1)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        num_points = points.shape[0]
        itemsize = np.dtype(np.float32).itemsize
        point_element_size = 3*itemsize + 3
        byte_array = np.zeros([num_points, point_element_size], dtype=np.byte)
        byte_array[:,:12] = np.frombuffer(points.tobytes(), dtype=np.byte).reshape(num_points, 3*4)
        byte_array[:, 12:] = np.frombuffer( colors.tobytes(), dtype=np.byte).reshape(num_points, 3)
        pcd_msg = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=point_element_size,
            row_step=(itemsize * 3 * points.shape[0]),
            data=byte_array.tobytes()
        )
        self.pcd_publisher.publish(pcd_msg)

    def sync_cb(self, left_img_msg, right_img_msg, left_info_msg, right_info_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding="rgb8")
        right_img = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding="rgb8")
        left_tensor_img = self.move_image_to_gpu(left_img)
        right_tensor_img = self.move_image_to_gpu(right_img)
        padder = InputPadder(left_tensor_img.shape, divis_by=32, force_square=False)
        pad_left_img, pad_right_img = padder.pad(left_tensor_img, right_tensor_img)
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            gpu_padded_disparity_map = self.model.forward(
                pad_left_img, pad_right_img, iters=4, test_mode=True,
                optimize_build_volume='pytorch1')
        gpu_disparity_map = padder.unpad(gpu_padded_disparity_map.float())[0,0]
        disparity_map = gpu_disparity_map.data.cpu().numpy().clip(0, None)
        self.publish_disparity_image(disparity_map, left_img_msg.header)
        self.publish_point_cloud(disparity_map, left_img,
            left_info_msg, right_info_msg, left_img_msg.header)


torch.autograd.set_grad_enabled(False)
rclpy.init()
ffs_node = FFSNode()
rclpy.spin(ffs_node)
ffs_node.destroy_node()
rclpy.shutdown()
