import copy
import atexit
import random
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import slam_utils


class SLAM:
    def __init__(self, filename):
        self.num_particles = 24
        self.num_angles = 360
        self.probs = np.zeros([self.num_particles])
        data = np.load(filename)
        self.odom = data["odom"]
        self.scans = data["scans"]
        self.particles = np.tile(np.array([0.0, 0.0, -0.5 * np.pi]), reps=(self.num_particles, self.odom.shape[0], 1))
        # shape N, T, P where N is particle no, T is time, P is pose shape

    def rollout(self):
        self.particles = np.tile(np.array([0.0, 0.0, -0.5 * np.pi]), reps=(self.num_particles, self.odom.shape[0], 1))
        for t in range(1, self.odom.shape[0]):
            self.particles[:, t] = slam_utils.sample_motion_model_odometry(self.particles[:, t-1], self.odom[t-1])
            predictions = self.laser_pred(t)
            logprobs_particles = slam_utils.laser_probs(predictions, self.scans[t])
            probs = np.exp(logprobs_particles)
            norm_probs = probs/probs.sum()
            self.particles[:, t] = self.resample_particles(self.particles[:,t], norm_probs)

    def update(self, scan, robot_frame_odom):
        particles = slam_utils.sample_motion_model_odometry(robot_frame_odom)
        self.odom_delta.append(robot_frame_odom)

    def resample_particles(self, particles, probs):
        resampled_particle_indices = np.random.choice(np.arange(self.num_particles),
size=self.num_particles, p=probs)
        resampled_particles = particles[resampled_particle_indices]
        return resampled_particles

    def likelihood(self):
        likelihood = np.zeros([self.num_particles])
        for t in range(1, self.odom.shape[1]):
            predictions = self.laser_pred(t)
            logprobs_particles = slam_utils.laser_probs(predictions, self.scans[t])
            likelihood += logprobs_particles
        return likelihood

    def laser_pred(self, t):
        predictions = np.zeros([self.num_particles, 360])
        for p in range(self.num_particles):
            rel_node = self.select(p, self.particles[p, t], t)
            predictions[p] = slam_utils.pred(self.scans[rel_node], self.particles[p, rel_node], self.particles[p, t])
        return predictions

    def interior(self, scan, scan_pose, query_pose):
        xp = scan_pose[0] - query_pose[0]
        yp = scan_pose[1] - query_pose[1]
        R = np.sqrt(xp*xp + yp*yp)
        if R < .5:
            return 1
        else:
            return 0

    def select(self, particle_idx, particle, pt):  # We discretise pose to 1m and ask for pose closest to this
        min_dist = 10000
        for t in range(pt):
            dist = np.sqrt( (self.particles[particle_idx, t, 0]-int(particle[0]))**2 + (self.particles[particle_idx, t, 1] - int(particle[1]))**2)
            if dist < min_dist:
                min_dist = dist
                idx = t
        return idx

    def create_map(self, particle_idx):
        # Returns indices of mapping poses, assuming particle particle_idx
        ref_poses = []
        positions = np.unique(self.particles[particle_idx, :,:2].astype(np.int32), axis=0)
        for p in range(positions.shape[0]):
            ref = self.select(particle_idx, positions[p], self.particles.shape[1])
            ref_poses.append(ref)
        return ref_poses


class SLAMNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.view_publisher = self.create_publisher(Marker, '/view_marker', 1)
        self.path_publisher = self.create_publisher(Marker, '/path_marker', 1)
        self.declare_parameter('filename', 'odom.npz')
        filename = self.get_parameter('filename').get_parameter_value().string_value
        self.declare_parameter('map_filename', 'map_odom.npz')
        self.map_filename = self.get_parameter('map_filename').get_parameter_value().string_value
        self.slam = SLAM(filename)
        self.colors = [ [random.random(), random.random(), random.random(), 1.0] for i in range(100)]

    def create_view(self, pose, scan):
        # from a scan associated with a pose, produce associated points in ROS2 coords.
        mylist = []
        for b in range(360):
            x = np.cos(b*2*np.pi/360 + pose[2])*scan[b] + pose[0]
            y = np.sin(b*2*np.pi/360 + pose[2])*scan[b] + pose[1]
            if np.isnan(scan[b]):
                pass
            else:
                mylist.append([x,y])
        return mylist

    def publish_points(self, flat_points, flat_colors):
        # List of all points and colors describing an effective map
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "0"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = 0.0, 0.0, 0.0
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = 0.0, 0.0, 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = 0.03, 0.03, 0.05
        marker.points = [Point(x=x,y=y) for (x,y) in flat_points]
        marker.colors = [ColorRGBA(r=r, g=g, b=b, a=a) for (r,g,b,a) in flat_colors]
        marker.frame_locked = True
        self.view_publisher.publish(marker)

    def publish_path(self, poses):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "0"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = 0.0, 0.0, 0.0
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z = 0.0, 0.0, 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = 0.03, 0.03, 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.points = [Point(x=x,y=y) for (x,y,theta) in poses]
        marker.frame_locked = True
        self.path_publisher.publish(marker)

    def publish_map(self, particle_idx):
        flat_points = []
        flat_colors = []
        refs = self.slam.create_map(particle_idx)
        for idx in range(len(refs)):
            print("Pos=", idx)
            points = self.create_view(self.slam.particles[particle_idx, refs[idx]], self.slam.scans[refs[idx]])
            for point in points:
                flat_points.append(point)
                flat_colors.append(self.colors[idx])
        self.publish_points(flat_points, flat_colors)

    def run(self):
        best_prob = -1000000
        while True:
            self.slam.rollout()
            prob = self.slam.likelihood()
            est_prob = prob.max()
            idx = prob.argmax()
            if est_prob > best_prob:
                best_prob = est_prob
                self.best_particle = idx
                self.publish_path(self.slam.particles[self.best_particle])
                self.publish_map(idx)
                print(self.best_particle, best_prob)

    def exit(self):
        print("Exiting")
        refs = self.slam.create_map(self.best_particle)
        np.savez(self.map_filename, poses=np.array(self.slam.particles[self.best_particle, refs]), scans=np.array(self.slam.scans[refs]))
        self.destroy_node()


rclpy.init()
slam_node = SLAMNode()
atexit.register(slam_node.exit)
slam_node.run()
