import random
import skimage
import numpy as np
from scipy.stats import vonmises
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped, PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros.transform_listener import TransformListener
from tf_transformations import euler_from_quaternion
from tf2_ros.buffer import Buffer
# Current StaticTransformBroadcaster is broken, we need to use from rolling.
# clone git clone https://github.com/ros2/geometry2.git
# Prepend ./src/geometry2/tf2_ros_py/tf2_ros to PYTHONPATH and export
from static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformException, LookupException, ExtrapolationException
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import slam_utils




def trobot_frame_odom(previous_odom_pose, current_odom_pose):
    diff_x = current_odom_pose[0] - previous_odom_pose[0]
    diff_y = current_odom_pose[1] - previous_odom_pose[1]
    forward = diff_x * np.cos(previous_odom_pose[2]) + diff_y * np.sin(previous_odom_pose[2])
    slip = diff_x * np.cos(previous_odom_pose[2] + np.pi/2) + diff_y * np.sin(previous_odom_pose[2] + np.pi/2)
    diff_angle = slam_utils.angle_diff(current_odom_pose[2], previous_odom_pose[2])
    d_rot = current_odom_pose[2] - previous_odom_pose[2]
    print("forward=", forward, " slip=", slip, d_rot)
    return forward, slip, d_rot


class Localizer:
    def __init__(self, filename):
        self.num_particles = 2000
        data = np.load(filename)
        self.views_poses = data["poses"]
        self.views_scans = data["scans"]
        self.particles = self.generate_interior_particles(self.num_particles)
        self.best_match = -100000.0

    def update_from_lidar(self, scan, magic_corr):
        kidnap_particles_num = int(self.num_particles*.1)
        particles = np.zeros([self.num_particles + kidnap_particles_num, 3])
        particles[:self.num_particles] = self.particles
        particles[self.num_particles:] = self.generate_interior_particles(kidnap_particles_num)
        predictions = self.lidar_range_predictions(particles)
        logprobs_particles = slam_utils.logprob_range_predictions(predictions, scan)
        self.best_match = logprobs_particles.max()
        print("MAX PROB=", self.best_match)
        # We remove exterior as our random odometry may have put us outside
        self.remove_exterior(logprobs_particles, scan)
        probs = np.exp(logprobs_particles/magic_corr)
        norm_probs = probs/probs.sum()
        self.particles = self.resample_particles(particles, norm_probs, self.num_particles)

    def lost(self):
        if self.best_match < -900:
            return True
        else:
            return False

    def set_pose(self, x, y, angle):
        self.particles = np.zeros([self.num_particles, 3]) + np.array([x, y, angle])

    def generate_interior_particles(self, num_particles):
        particles = []
        while len(particles) < num_particles:
            view = np.random.randint(0, len(self.views_poses))
            angle = np.random.randint(0, 360)
            if np.isnan(self.views_scans[view][angle]):
                continue
            sample_range = np.random.uniform(low=0, high=self.views_scans[view][angle])
            xp = sample_range * np.cos(self.views_poses[view][2] + np.pi * 2*angle/360) + self.views_poses[view][0]
            yp = sample_range * np.sin(self.views_poses[view][2] + np.pi * 2*angle/360) + self.views_poses[view][1]
            sample_orientation = np.random.uniform(0, 360)
            particles.append(np.array([xp, yp, sample_orientation]))
        return np.array(particles)

    def remove_exterior(self, logprobs_particles, scan):
        for p in range(self.num_particles):
            idx = self.find_closest_view(self.particles[p])
            if not self.is_interior(scan, self.views_poses[idx], self.particles[p]):
                logprobs_particles[p] = -10000

    def lidar_range_predictions(self, particles):
        num_particles = particles.shape[0]
        predictions = np.zeros([num_particles, 360])
        for p in range(num_particles):
            rel_node = self.find_closest_view(particles[p])
            predictions[p] = slam_utils.pred(self.views_scans[rel_node], self.views_poses[rel_node], particles[p])
        return predictions

    def find_closest_view(self, particle):  # We discretise pose to 1m and ask for pose closest to this
        min_dist = 10000
        for t in range(len(self.views_poses)):
            dist = np.sqrt( (self.views_poses[t, 0]-int(particle[0]))**2 + (self.views_poses[t, 1] - int(particle[1]))**2)
            if dist < min_dist:
                min_dist = dist
                idx = t
        return idx

    def is_interior(self, scan, scan_pose, query_pose):
        xp = query_pose[0] - scan_pose[0]
        yp = query_pose[1] - scan_pose[1]
        angle = np.arctan2(yp, xp)
        R = np.sqrt(xp*xp + yp*yp)
        rolled_scan = np.roll(scan, int((-scan_pose[2]+np.pi/2)*360/(2*np.pi)))
        if rolled_scan[int(angle)] > R:
            return True
        else:
            return False

    def resample_particles(self, particles, probs, num_resampled_particles):
        num_particles = particles.shape[0]
        resampled_particle_indices = np.random.choice(np.arange(num_particles),
            size=num_resampled_particles, p=probs)
        resampled_particles = particles[resampled_particle_indices]
        return resampled_particles

    def expected_pose(self):
        x_mean, y_mean, _ = np.mean(self.particles, axis=0)
        _, angle, _ = vonmises.fit(self.particles[:, 2], fscale=1)
        return x_mean, y_mean, angle


class LocalizerNode(Node):
    def __init__(self):
        super().__init__("localizer_node")
        self.min_dist = 0.03  # minimum distance for lidar update
        self.min_angle = .08  # minimum angle change for lidar update
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            1)
        self.initialpose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.initialpose_callback,
            1)
        self.view_publisher = self.create_publisher(Marker, '/view_marker', 1)
        self.particles_publisher = self.create_publisher(Marker, '/particles', 1)
        self.view_publisher.publish(Marker())
        self.tf_buffer = Buffer()
        qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            )
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True,
qos=qos)
        self.current_lidar_msg = None
        self.previous_odom_pose = None
        self.init_wait = 0
        self.declare_parameter('map_filename', 'map_odom.npz')
        map_filename = self.get_parameter('map_filename').get_parameter_value().string_value
        self.localizer = Localizer(map_filename)
        self.colors = [ [random.random(), random.random(), random.random(), 1.0] for
i in range(100)]
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_map()
        self.initial_pose_received = False

    def initialpose_callback(self, initialpose_msg):
        try:
            tf_base_link_to_base_laser = self.tf_buffer.lookup_transform(
                "base_link",
                "base_laser",
                Time())
        except TransformException as ex:
            warn_msg = "No Transform for base_link to base_laser. Initial pose not set."
            self.get_logger().warn(warn_msg)
            return
        self.initial_pose_received = True
        r_t = initialpose_msg.pose.pose.orientation
        rot = [r_t.x, r_t.y, r_t.z, r_t.w]
        _, _, theta_laser = euler_from_quaternion(rot)
        # I am assuming here that laser and base_link just differ by orientation
        _, _, theta_diff = self.ros2_to_pose(tf_base_link_to_base_laser)
        self.localizer.set_pose(initialpose_msg.pose.pose.position.x, initialpose_msg.pose.pose.position.y, theta_laser + theta_diff)
        self.get_logger().info("Initial pose set.")

    def publish_particles(self, pose):
        marker = Marker()
        marker.header.stamp = self.current_lidar_msg.header.stamp
        marker.header.frame_id = "base_laser"
        marker.ns = "basic_shapes"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = 0.0, 0.0, 0.0
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = 0.0, 0.0, 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = 0.03, 0.03, 0.05
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.3, 1.0, 1.0, .2
        particles_base_laser = np.matmul(self.localizer.particles[:, :2] - pose[:2], R.from_rotvec([0, 0, pose[2]]).as_matrix()[:2, :2])
        marker.points = [Point(x=x,y=y) for (x, y) in particles_base_laser.tolist()]
        marker.frame_locked = True
        self.particles_publisher.publish(marker)

    def publish_map(self):
        flat_points = []
        flat_colors = []
        for idx in range(len(self.localizer.views_poses)):
            points = slam_utils.create_view(self.localizer.views_poses[idx], self.localizer.views_scans[idx])
            for point in points:
                flat_points.append(point)
                flat_colors.append(self.colors[idx])
        self.view_publisher.publish(slam_utils.publish_points(self.view_publisher, self.get_clock().now().to_msg(), flat_points, flat_colors))


    def publish_map_odom_transform(self, tf_base_laser_to_odom, pose):
        tf_zero_to_odom = TransformStamped()
        tf_zero_to_odom.header.stamp = self.current_lidar_msg.header.stamp
        tf_zero_to_odom.header.frame_id = 'zero'
        tf_zero_to_odom.child_frame_id = 'odom'
        tf_zero_to_odom.transform = tf_base_laser_to_odom.transform
        tf_map_to_zero = TransformStamped()
        tf_map_to_zero.header.stamp = self.current_lidar_msg.header.stamp
        tf_map_to_zero.header.frame_id = 'map'
        tf_map_to_zero.child_frame_id = 'zero'
        tf_m_to_z_trans = tf_map_to_zero.transform.translation
        tf_m_to_z_trans.x, tf_m_to_z_trans.y, tf_m_to_z_trans.z = pose[0], pose[1], 0.0
        q = quaternion_from_euler(0, 0, pose[2])
        tf_m_to_z_rot = tf_map_to_zero.transform.rotation
        tf_m_to_z_rot.x, tf_m_to_z_rot.y, tf_m_to_z_rot.z, tf_m_to_z_rot.w = q[0], q[1], q[2], q[3]
        self.tf_static_broadcaster.sendTransform([tf_zero_to_odom, tf_map_to_zero])

    def robot_frame_odom(self, previous_odom_pose, current_odom_pose):
        diff_x = current_odom_pose[0] - previous_odom_pose[0]
        diff_y = current_odom_pose[1] - previous_odom_pose[1]
        forward = diff_x * np.cos(previous_odom_pose[2]) + diff_y * np.sin(previous_odom_pose[2])
        slip = diff_x * np.cos(previous_odom_pose[2] + np.pi/2) + diff_y * np.sin(previous_odom_pose[2] + np.pi/2)
        diff_angle = slam_utils.angle_diff(current_odom_pose[2], previous_odom_pose[2])
        d_rot = current_odom_pose[2] - previous_odom_pose[2]
        print("forward=", forward, " slip=", slip, d_rot)
        return forward, slip, d_rot

    def ros2_to_pose(self, odom_transform):
        trans_tf = odom_transform.transform.translation
        rot_tf = odom_transform.transform.rotation
        rot = [rot_tf.x, rot_tf.y, rot_tf.z, rot_tf.w]
        _, _, theta = euler_from_quaternion(rot)
        return (trans_tf.x, trans_tf.y, theta)

    def robot_moved(self, current_odom_pose):
        diff_x = current_odom_pose[0] - self.previous_odom_pose[0]
        diff_y = current_odom_pose[1] - self.previous_odom_pose[1]
        d_trans = np.sqrt(diff_y**2 + diff_x**2)
        diff_angle = slam_utils.angle_diff(current_odom_pose[2], self.previous_odom_pose[2])
        return np.abs(diff_angle) > self.min_angle or d_trans > self.min_dist

    def lidar_callback(self, lidar_msg):
        # Filters lidar msg so only those with a valid base_laser <-> odom
        # transform are passed on to process_lidar
        self.publish_map()
        if self.current_lidar_msg is None:
            self.current_lidar_msg = lidar_msg
            return
        lidar_msg_time = Time.from_msg(self.current_lidar_msg.header.stamp)
        try:
            tf_base_laser_to_odom = self.tf_buffer.lookup_transform(
                "base_laser",
                "odom",
                lidar_msg_time)  # https://github.com/ros2/ros2_documentation/issues/4385
            tf_odom_to_base_laser = self.tf_buffer.lookup_transform(
                "odom",
                "base_laser",
                lidar_msg_time)
        except LookupException as ex:
            return
        except ExtrapolationException as ex:  # This is common and normal.
            # if our lidar_msg is before any transform, we will never be able to
            # transform it
            if "past" in str(ex):
                self.current_lidar_msg = None
                return
            return
        self.process_lidar(self.current_lidar_msg, tf_base_laser_to_odom, tf_odom_to_base_laser)
        self.current_lidar_msg = None

    def process_lidar(self, lidar_msg, tf_base_laser_to_odom, tf_odom_to_base_laser):
        if not(self.initial_pose_received):
            return
        scan = skimage.transform.resize(np.array(lidar_msg.ranges).astype(np.float32), (360,))
        current_odom_pose = self.ros2_to_pose(tf_odom_to_base_laser)
        print("S=", self.localizer.best_match)
        if self.localizer.lost() and False: # tmp disable
            self.localizer.particles += np.random.normal(size=[self.localizer.num_particles, 3])*.05
            self.localizer.update_from_lidar(scan, 1500)
            pose = self.localizer.expected_pose()
            print("LOST")
            self.publish_particles(pose)
            self.publish_map_odom_transform(tf_base_laser_to_odom, pose)
            self.previous_odom_pose = current_odom_pose
            return
        if self.previous_odom_pose is None:
            self.previous_odom_pose = current_odom_pose
        if self.robot_moved(current_odom_pose):
            odom = trobot_frame_odom(self.previous_odom_pose, current_odom_pose)
            self.localizer.particles = slam_utils.sample_motion_model_odometry(self.localizer.particles, odom)
            self.localizer.update_from_lidar(scan, 3600)
            robot_frame_odom = self.robot_frame_odom(self.previous_odom_pose, current_odom_pose)
            pose = self.localizer.expected_pose()
            print("pose=", pose)
            self.publish_particles(pose)
            self.publish_map_odom_transform(tf_base_laser_to_odom, pose)
            self.previous_odom_pose = current_odom_pose


rclpy.init()
localizer_node = LocalizerNode()
#localizer_node.publish_map()
rclpy.spin(localizer_node)
localizer_node.destroy_node()
rclpy.shutdown()

