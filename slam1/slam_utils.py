import numpy as np


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
