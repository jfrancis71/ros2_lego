import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
from visualization_msgs.msg import Marker
import slam_utils
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class StereoNode(Node):
    def __init__(self):
        super().__init__("mystereo_node")
        self.points_subscription = self.create_subscription(
            PointCloud2,
            "/points2",
#            "points2",
            self.ros2_points_callback,
            1)
        self.view_publisher = self.create_publisher(Marker, '/my_stereo_marker', 1)


    def publish_points(self, points, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "base_link"
        marker.ns = "0"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = 0.0, 0.0, 0.0
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = 0.0, 0.0, 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = 0.03, 0.03, 0.05
        #marker.points = [Point(x=x,y=y) for (x,y) in flat_points]
        marker.points = []
        for i in range(points.shape[0]):
            if points[i,1] < -.1 and points[i,1] > -.2:
                marker.points.append( Point(x=points[i,2].item(), y=-points[i,0].item()) )
#        marker.colors = [ColorRGBA(r=r, g=g, b=b, a=a) for (r,g,b,a) in flat_colors]
        marker.colors = [ColorRGBA(r=.8, g=.3, b=.5, a=1.0) for _ in range(len(marker.points))]
        marker.frame_locked = True
        self.view_publisher.publish(marker)

    def ffs_points_callback(self, points_msg):
        num_points = points_msg.width
        itemsize = np.dtype(np.float32).itemsize
        point_element_size = 3*itemsize + 3
        raw = np.ndarray(
            shape=(points_msg.width * points_msg.height, 15),
            dtype=np.byte,
            buffer=points_msg.data)
        my_raw = np.array(raw[:, :12])
        print("RAW=", points_msg.data[:12])
        print("size=", my_raw.nbytes)
        pt = np.ndarray(
                shape=(num_points, 3),
                dtype=np.float32,
                buffer=my_raw)
        print("Leg=", len(points_msg.data))
        print("Callback", pt[:5])
        self.publish_points(pt, points_msg.header.stamp)


    def ros2_points_callback(self, points_msg):
        num_points = points_msg.width*points_msg.height
        itemsize = np.dtype(np.float32).itemsize
        point_element_size = 3*itemsize + 3
        raw = np.ndarray(
            shape=(points_msg.width * points_msg.height, points_msg.point_step),
            dtype=np.byte,
            buffer=points_msg.data)
        my_raw = np.array(raw[:, :12])
        print("size=", my_raw.nbytes)
        pt = np.ndarray(
                shape=(num_points, 3),
#                shape=(240*320, 3),
                dtype=np.float32,
                buffer=my_raw)
        print("Leg=", len(points_msg.data))
        print("Callback", pt[:5])
        print("pt=", pt)
        self.publish_points(pt, points_msg.header.stamp)

rclpy.init()
stereo_node = StereoNode()
rclpy.spin(stereo_node)
rclpy.shutdown()
