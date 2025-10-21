import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
import skimage
import numpy as np
import itertools

class Nav(Node):
    def __init__(self):
        super().__init__("nav")
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.pred_publisher = \
            self.create_publisher(LaserScan, "/pred_laser", 10)
        self.house = skimage.io.imread("~/ros2_ws/my_house.pgm")
        self.centers = [(78, 200), (80, 202), (80, 200), (80, 198), (80, 196), (82, 200)]
        self.centers = sum([[(w, c) for c in range(196,200)] for w in range(78,82)], [])
        self.centers = [(70, 200)]
        self.centers = [(30, 220)]
        self.centers = [(30, 220), (40, 220)]
#        self.centers = [(80,200), (90, 200), (100, 200)]
        self.centers = [(80, 200), (80, 190), (80, 180)]
        self.centers = [[(r,c) for c in range(150,220,10)] for r in range(40,100,10)]
        self.centers = list(itertools.chain(*self.centers))

        self.trans = np.array([skimage.transform.warp_polar(self.house==0, center=c, radius=100) for c in self.centers])
        self.polar_coords = np.array([np.argmax(self.trans[c], axis=1)*0.05 for c in range(len(self.centers))])
        #self.trans = np.flip(np.array([skimage.transform.warp_polar(self.house==0, center=c, radius=100) for c in self.centers]), axis=1)
        self.tf_broadcaster = TransformBroadcaster(self)

    def lidar_callback(self, lidar_msg):
#        print("Received lidar message.", lidar_msg.header.frame_id)
#        print("Message length", len(lidar_msg.ranges))
        scan = np.array(lidar_msg.ranges)
        new_scan = skimage.transform.resize(scan.astype(np.float32), (360,))
#        predictions = np.roll(np.array([[np.flip(np.argmax(self.trans[c], axis=1)*0.05) for s in range(359)] for c in range(len(self.centers))]), 90, axis=2)
        predictions = np.array([[np.roll(np.flip(self.polar_coords[c]), 90+s) for s in range(359)] for c in range(len(self.centers))])
        print("pred shape=", predictions.shape)

#        angles = np.nanmean((predictions - new_scan)**2, axis=0)[np.newaxis, np.newaxis]
        prediction_error = np.nanmean((predictions - new_scan)**2, axis=2)
        idx = np.unravel_index(np.argmin(prediction_error), prediction_error.shape)
#        idx = [0,0]
        print("idx=", idx)
        loc = idx[0]
        angle = 2 * np.pi * idx[1]/360.0
        print("Best =", loc, " match=", prediction_error[idx])

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = self.centers[loc][1]*.05 -13.640
        t.transform.translation.y = (118-self.centers[loc][0])*0.05 -3.959
        t.transform.translation.z = 0.0

        q = quaternion_from_euler(0, 0, -angle)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

        lidar_msg1 = LaserScan()
        lidar_msg1.ranges = predictions[idx[0], idx[1]]
        lidar_msg1.angle_min = 0.0
        lidar_msg1.angle_max = 6.28318548
        lidar_msg1.angle_increment = 2 * np.pi / 360.
        lidar_msg1.time_increment = 0.00019850002718158066
        lidar_msg1.scan_time = 0.10004401206970215
        lidar_msg1.range_min = 0.019999999552965164
        lidar_msg1.range_max = 25.0
        #lidar_msg.header
        lidar_msg1.header = lidar_msg.header
        self.pred_publisher.publish(lidar_msg1)




rclpy.init()
nav = Nav()
rclpy.spin(nav)
nav.destroy_node()
rclpy.shutdown()
