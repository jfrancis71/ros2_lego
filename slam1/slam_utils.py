import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


def create_view(pose, scan):
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


def publish_points(publisher, t, flat_points, flat_colors):
    # List of all points and colors describing an effective map
    marker = Marker()
    marker.header.stamp = t
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
    print("PUBLISIHNG", marker.points[:10])
    return marker


def sample_motion_model_odometry(last_particles, robot_frame_odom):
    # Loosely based on p136 of Probabilistic Robotics
    num_particles = last_particles.shape[0]
    particles = np.zeros([num_particles, 3])
    sp1 = np.random.normal(size=num_particles)
    sp2 = np.random.normal(size=num_particles)
    sp3 = np.random.normal(size=num_particles)
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
