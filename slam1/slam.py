import copy
import atexit
import random
import numpy as np
from scipy.stats import norm
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


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
            self.particles[:, t] = self.extend(self.particles[:, t-1], self.odom[t-1])
            predictions = self.laser_pred(t)
            logprobs_particles = self.laser_probs(predictions, self.scans[t])
            probs = np.exp(logprobs_particles)
            norm_probs = probs/probs.sum()
            self.particles[:, t] = self.resample_particles(self.particles[:,t], norm_probs)

    def update(self, scan, robot_frame_odom):
        particles = self.extend(robot_frame_odom)
        self.odom_delta.append(robot_frame_odom)

    def resample_particles(self, particles, probs):
        resampled_particle_indices = np.random.choice(np.arange(self.num_particles),
size=self.num_particles, p=probs)
        resampled_particles = particles[resampled_particle_indices]
        return resampled_particles

    def extend(self, last_particles, robot_frame_odom):
        particles = np.zeros([self.num_particles, 3])
        sp1 = np.random.normal(size=self.num_particles)
        sp2 = np.random.normal(size=self.num_particles)
        sp3 = np.random.normal(size=self.num_particles)
        forward = robot_frame_odom[0]
        sample_forward = forward + .1*sp1*forward
        slide = robot_frame_odom[1]
        sample_slide = slide + .1*sp2*slide
        spin = robot_frame_odom[2]
        sample_spin = spin + .1*sp3*spin
        particles[:, 0] = last_particles[:, 0] + sample_forward * np.cos(last_particles[:, 2]) + sample_slide * np.cos(last_particles[:, 2] + np.pi/2)
        particles[:, 1] = last_particles[:, 1] + sample_forward * np.sin(last_particles[:, 2]) + sample_slide * np.sin(last_particles[:, 2] + np.pi/2)
        particles[:, 2] = last_particles[:, 2] + sample_spin
#        particles[:, 0] = last_particles[:, 0] + sp1*.1
#        particles[:, 1] = last_particles[:, 1] + sp2*.1
#        particles[:, 2] = last_particles[:, 2] + sp3*.2

        return particles

    def likelihood(self):
        likelihood = np.zeros([self.num_particles])
        for t in range(1, self.odom.shape[1]):
            predictions = self.laser_pred(t)
            logprobs_particles = self.laser_probs(predictions, self.scans[t])
            likelihood += logprobs_particles
        return likelihood

    def laser_pred(self, t):
        predictions = np.zeros([self.num_particles, 360])
        for p in range(self.num_particles):
            rel_node = self.select(p, self.particles[p, t], t)
            predictions[p] = self.pred(self.scans[rel_node], self.particles[p, rel_node], self.particles[p, t])
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

    # Computes range predictions from a view pose to a query pose
    def pred(self, scan, scan_pose, query_pose):
        ranges = np.zeros([360]) * np.nan
        linespace = np.arange(360)
        x = scan_pose[0] + scan * np.cos(linespace*2*np.pi/360 + scan_pose[2])
        y = scan_pose[1] + scan * np.sin(linespace*2*np.pi/360 + scan_pose[2])
        xp = x - query_pose[0]
        yp = y - query_pose[1]
        na = query_pose[2]
        X = xp * np.cos(na) + yp * np.sin(na)
        Y = -xp * np.sin(na) + yp * np.cos(na)
        R = np.sqrt(X*X + Y*Y)
        THETA = np.arctan2(Y, X)
        for a in range(360):
            if np.isnan(THETA[a]):
                continue
            idx = int(THETA[a]*360/(2*np.pi)) % 360
            if np.isnan(ranges[idx]):
                ranges[idx] = R[a]
            else:
                if R[a] < ranges[idx]:
                    ranges[idx] = R[a]
        return ranges

    def laser_probs(self, predictions, scan):
        probs = np.zeros([self.num_particles])
        for p in range(self.num_particles):
            probs[p] = np.nanmean(norm.logpdf(scan, loc=predictions[p], scale=0.1))
        return probs/1000


class SLAMNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.view_publisher = self.create_publisher(Marker, '/view_marker', 1)
        self.path_publisher = self.create_publisher(Marker, '/path_marker', 1)
        self.declare_parameter('filename', 'odom.npz')
        filename = self.get_parameter('filename').get_parameter_value().string_value
        self.slam = SLAM(filename)
        self.colors = [ [random.random(), random.random(), random.random(), 1.0] for i in range(100)]

    def create_points(self, particle, scan):
        mylist = []
        for b in range(360):
            x = np.cos(b*2*np.pi/360 + particle[2])*scan[b] + particle[0]
            y = np.sin(b*2*np.pi/360 + particle[2])*scan[b] + particle[1]
            if np.isnan(scan[b]):
                pass
            else:
                mylist.append([x,y])
        return mylist

    def publish_view(self, flat_points, flat_colors):
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

    def particle_info(self, particle_idx):
        positions = np.unique(self.slam.particles[particle_idx, :,:2].astype(np.int32), axis=0)
        flat_points = []
        flat_colors = []
        for p in range(positions.shape[0]):
            ref = self.slam.select(particle_idx, positions[p], self.slam.particles.shape[1])
            print("Pos=", positions[p], ref)
            points = self.create_points(self.slam.particles[particle_idx, ref], self.slam.scans[ref])
            for point in points:
                flat_points.append(point)
                flat_colors.append(self.colors[p])
        self.publish_view(flat_points, flat_colors)

    def run(self):
        best_prob = -1000000
        while True:
            self.slam.rollout()
            prob = self.slam.likelihood()
            est_prob = prob.max()
            idx = prob.argmax()
            if est_prob > best_prob:
                best_prob = est_prob
                best_particle = self.slam.particles[idx].copy()
                self.publish_path(best_particle)
                self.particle_info(idx)
                print(best_particle, best_prob)

    def exit(self):
        print("Exiting")
        self.destroy_node()


rclpy.init()
slam_node = SLAMNode()
atexit.register(slam_node.exit)
slam_node.run()
