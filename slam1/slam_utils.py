import numpy as np
from scipy.stats import norm
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


def angle_diff(angle_1, angle_2):
    """Returns closest difference between two angles.

       Note you cannot just subtract the angles, eg 3.1 - (-3.1) = 6.2. This not the
closest angle change.

    """
    abs_diff_angle = np.abs(angle_1 - angle_2)
    return np.min(np.array([abs_diff_angle, 2*np.pi - abs_diff_angle]))


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


# Computes range predictions from a view pose to a query pose
def pred(scan, scan_pose, query_pose):
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


def laser_probs(predictions, scan):
    num_particles = predictions.shape[0]     
    probs = np.zeros([num_particles])
    for p in range(num_particles):
        probs[p] = np.nanmean(norm.logpdf(scan, loc=predictions[p], scale=0.1))
    return probs/1000

