import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
import numpy as np
import time


class Nav(Node):
    def __init__(self):
        super().__init__("nav")
        np.set_printoptions(edgeitems=20, linewidth=200)
        self.base_ranges = None

        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.occ_publisher = self.create_publisher(OccupancyGrid, "/map", 10)
        self.pose_publisher = \
            self.create_publisher(PoseStamped, "nav_pose", 10)

    def convert_to_2d(self, lidar_msg):
        points = np.zeros([len(lidar_msg.ranges), 2])
        for l in range(len(lidar_msg.ranges)):
            angle = l*2*np.pi/500
            distance = lidar_msg.ranges[l]
            if np.isnan(distance):
                continue
            points[l, 0] = distance*np.sin(angle)
            points[l, 1]  = distance*np.cos(angle)
        return points

    def create_message(self, lidar_msg):
        msg = OccupancyGrid()
        time.sleep(1)
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_laser"
        msg.info.resolution = 0.1
        msg.info.width = msg.info.height = 60
        msg.info.origin.position.x = -3.0
        msg.info.origin.position.y = -3.0
        msg.info.origin.position.z = 0.0
        map = np.zeros([60, 60], dtype=np.int32)
        for l in range(len(lidar_msg.ranges)):
            angle = l*2*np.pi/500
            distance = lidar_msg.ranges[l]
            if np.isnan(distance):
                continue
            row_idx = np.clip(int(30+10*distance*np.sin(angle)), 0, 59)
            col_idx = np.clip(int(30+10*distance*np.cos(angle)), 0, 59)
            map[row_idx, col_idx] = 100
        my_list = map.flatten().tolist()
        #print(map)
        msg.data = my_list
        return msg

    def publish_pose(self, header, x, y):
        pose_msg = Pose()
        point = Point()
        point.x = x
        point.y = y
        pose_msg.position = point
        orientation = 0.0
        quaternion = R.from_euler('xyz', [0, 0, orientation*2*3.141]).as_quat()
        q = Quaternion()
        q.x, q.y, q.z, q.w = quaternion
        pose_msg.orientation = q
        pose_stamped = PoseStamped()
        pose_stamped.header = header
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose = pose_msg
        self.pose_publisher.publish(pose_stamped)


    def lidar_callback(self, lidar_msg):
        print("Received lidar message.")
        print("Message length", len(lidar_msg.ranges))
        if self.base_ranges is None:
            self.base_ranges = lidar_msg.ranges
            self.map_points = self.convert_to_2d(lidar_msg)
            msg = self.create_message(lidar_msg)
            self.occ_publisher.publish(msg)
            return
        new_points = self.convert_to_2d(lidar_msg)
        prob = np.zeros([21])
        for x in range(-10,10):
            dists = self.map_points[:500] - np.array([x/10, 0.0]) - new_points[:500]
            t = np.sqrt((dists*dists).sum())
            prob[x+10] = t
        f = prob[:20].argmin()
        print(prob/100, "min=", f)
        self.publish_pose(lidar_msg.header, (f-10)/10, 0.0)


rclpy.init()
nav = Nav()
rclpy.spin(nav)
nav.destroy_node()
rclpy.shutdown()
