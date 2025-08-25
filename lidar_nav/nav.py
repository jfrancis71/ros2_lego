import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from laser_geometry.laser_geometry import LaserProjection
from sensor_msgs.msg import PointCloud2
import numpy as np
import time
import math


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
        self.pose_publisher = \
            self.create_publisher(PoseStamped, "nav_pose", 10)
        self.base_publisher = \
            self.create_publisher(LaserScan, "/base_laser", 10)
        self.current_publisher = \
            self.create_publisher(LaserScan, "/current_laser", 10)
        self.trans_publisher = \
            self.create_publisher(LaserScan, "/trans_laser", 10)

        self.num_rand_poses = 1000
        self.rand_poses = np.random.rand(self.num_rand_poses, 3)
        self.rand_poses[:, :2] = self.rand_poses[:, :2] - 0.5
        self.rand_poses[:, 2] = self.rand_poses[:, 2] * 2 * math.pi
        self.laserProjection = LaserProjection()

    def convert_to_2d(self, lidar_msg):
        points = np.zeros([len(lidar_msg.ranges), 2])
        distances = np.array(lidar_msg.ranges)
        for l in range(len(lidar_msg.ranges)):
            angle = l*2*np.pi/500
            if np.isnan(distances[l]):
                continue
            points[l, 0] = distances[l]*np.cos(angle-math.pi/2)
            points[l, 1]  = distances[l]*np.sin(angle-math.pi/2)
        return points

    def publish_pose(self, header, x, y, theta):
        pose_msg = Pose()
        point = Point()
        point.x = x
        point.y = y
        pose_msg.position = point
        orientation = theta
        quaternion = R.from_euler('xyz', [0, 0, orientation]).as_quat()
        q = Quaternion()
        q.x, q.y, q.z, q.w = quaternion
        pose_msg.orientation = q
        pose_stamped = PoseStamped()
        pose_stamped.header = header
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose = pose_msg
        self.pose_publisher.publish(pose_stamped)

    def publish_laser(self, publisher, header, ranges, intensity):
        lidar_msg = LaserScan()
        lidar_msg.ranges = ranges[:500]
        lidar_msg.intensities = np.array(ranges[:500])*0.0 + intensity
        lidar_msg.angle_min = 0.0
        lidar_msg.angle_max = 6.28318548
        lidar_msg.angle_increment = 0.012466637417674065
        lidar_msg.time_increment = 0.00019850002718158066
        lidar_msg.scan_time = 0.10004401206970215
        lidar_msg.range_min = 0.019999999552965164
        lidar_msg.range_max = 25.0
        lidar_msg.header = header
        lidar_msg.header.frame_id = "base_laser"
        publisher.publish(lidar_msg)

    def transform(self, points, delta_x, delta_y, delta_theta):
        new_points1 = np.copy(points)
        new_points1[:, 0] -= delta_x
        new_points1[:, 1] -= delta_y
        new_points_x = points[:, 0] * np.cos(delta_theta) - points[:, 1] * np.sin(delta_theta) - delta_x
        new_points_y = points[:, 0] * np.sin(delta_theta) + points[:, 1] * np.cos(delta_theta) - delta_y
        new_points = np.stack([new_points_x, new_points_y], axis=1)
        return new_points

    def predict_ranges(self, points):  # points is in coordinate system of laser range finder.
        ranges = np.zeros([500])
        angles = np.arctan2(points[:, 1], points[:, 0])
        distances = np.linalg.norm(points, axis=1)
        for i in range(500):
            pred_angle = 2*math.pi * i/500
            point = np.argmin(np.cos(angles-pred_angle-math.pi/2))
            ranges[i] = distances[point]
        return ranges

    def predict_pose(self, lidar_ranges, lidar_msg):
        dists = np.zeros([self.num_rand_poses])
        for i in range(self.num_rand_poses):
            pose = self.rand_poses[i]
            trans_points = self.transform(self.map_points, pose[0], pose[1], pose[2])
#            trans_points = self.transform(self.map_points, pose[0], pose[1], 0.0)
            pred_ranges = self.predict_ranges(trans_points)
            dist = (lidar_ranges[:500] - pred_ranges)**2
            print("Dist=", i, np.nanmean(dist), self.rand_poses[i])
            dists[i] = np.nanmean(dist)
        best = np.argmin(dists)
        print("smallest = ", best, " pose=", self.rand_poses[best], "result=", dists[best])
        print("TEST")
        trans_points = self.transform(self.map_points, 0.0, 0.0, 0.0)
        pred_ranges = self.predict_ranges(trans_points)
        dist = (lidar_ranges[:500] - pred_ranges)**2
        print("Dist=", np.nanmean(dist), self.rand_poses[i])
        print("END")
        pose = self.rand_poses[best]
        trans_points = self.transform(self.map_points, pose[0], pose[1], pose[2])
#        trans_points = self.transform(self.map_points, 0.0, 0.0, pose[2])
        pred_ranges = self.predict_ranges(trans_points)
        self.publish_laser(self.trans_publisher, lidar_msg.header, pred_ranges, 200)
        return self.rand_poses[best, 0], self.rand_poses[best, 1], -self.rand_poses[best, 2]
       
    def lidar_callback(self, lidar_msg):
        print("Received lidar message.", lidar_msg.header.frame_id)
        print("Message length", len(lidar_msg.ranges))
        if self.base_ranges is None:
            self.base_ranges = lidar_msg.ranges
            self.map_points = self.convert_to_2d(lidar_msg)
            print(self.base_ranges)
            return
        new_points = self.convert_to_2d(lidar_msg)
        pose = self.predict_pose(lidar_msg.ranges, lidar_msg)
        self.publish_pose(lidar_msg.header, pose[0], pose[1], pose[2])
        self.publish_laser(self.base_publisher, lidar_msg.header, self.base_ranges, 100)
        self.publish_laser(self.current_publisher, lidar_msg.header, lidar_msg.ranges, 0)


rclpy.init()
nav = Nav()
rclpy.spin(nav)
nav.destroy_node()
rclpy.shutdown()
