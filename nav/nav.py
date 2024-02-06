from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
import torch


f = 428


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

    def listener_callback(self, msg):
        for det in msg.detections:
            print(det)
            dog_bottom_left_camera_x = 160-(det.bbox.center.position.x - det.bbox.size_x/2)
            dog_bottom_right_camera_x = 160-(det.bbox.center.position.x + det.bbox.size_x / 2)
            dog_x = 1.5
            dog_bottom_left_y = 0.14
            dog_bottom_right_y = -0.14
            dog_bottom_left_z = -.3  # check......

            xs = torch.linspace(-1.5, 1.5, steps=100)
            ys = torch.linspace(-1.5, 1.5, steps=100)
            thetas = torch.linspace(-1.5, 1.5, steps=16)
            x, y, theta = torch.meshgrid(xs, ys, thetas, indexing='xy')
            translate_1_y = dog_bottom_left_y-y
            translate_1_x = dog_x - x
            translate_2_y = dog_bottom_right_y - y
            translate_2_x = dog_x - x
            rotate_1_y = torch.cos(theta) * translate_1_y - torch.sin(theta) * translate_1_x
            rotate_1_x = torch.sin(theta) * translate_1_y  + torch.cos(theta) * translate_1_x
            rotate_2_y = torch.cos(theta) * translate_2_y - torch.sin(theta) * translate_2_x
            rotate_2_x = torch.sin(theta) * translate_2_y  + torch.cos(theta) * translate_2_x

            g1 = (rotate_1_y)/(rotate_1_x)
            g2 = (rotate_2_y) / (rotate_2_x)
            mypred1 = g1*f
            mypred2 = g2*f
            res1 = torch.distributions.normal.Normal(mypred1, 5).log_prob(torch.tensor(dog_bottom_left_camera_x))
            res2 = torch.distributions.normal.Normal(mypred2, 5).log_prob(torch.tensor(dog_bottom_right_camera_x))
            res = res1 + res2
            mynorm = res - torch.logsumexp(res, dim=[0, 1, 2], keepdim=False)
            smyprobs = torch.exp(mynorm)
            myprobs = torch.sum(smyprobs, axis=2)
#            myprobs = torch.ones([100,100])*0.5
            kernel = torch.ones([1, 1, 3, 3])
            conv = torch.nn.functional.conv2d(myprobs.unsqueeze(0), kernel, padding=1)[0]
            prob_map_msg = OccupancyGrid()
            prob_map_msg.header = msg.header
            prob_map_msg.header.frame_id = "base_link"
            prob_map_msg.info.resolution = 3/100
            prob_map_msg.info.width = 100
            prob_map_msg.info.height = 100
            prob_map_msg.info.origin.position.x = -1.5
            prob_map_msg.info.origin.position.y = -1.5
            prob_map_msg.data = (1000 * conv).type(torch.int).flatten().tolist()
            self.probmap_publisher.publish(prob_map_msg)


rclpy.init()
nav_node = Nav()
rclpy.spin(nav_node)
nav_node.destroy_node()
rclpy.shutdown()
