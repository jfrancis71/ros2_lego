# Lego
Lego is a ROS2 package for a lego Technic robot (Charlie) running ROS2 Humble from RoboStack (Conda installation)

![Charlie](https://drive.google.com/uc?id=1GdqDXQZsIsTLFUqVw2gdJ1S9lr9-x4DP&export=download)

## Components

### Robot:

Raspberry Pi 3 Model B+, Dexter Industries BrickPi3, EV3 Motors, HiTechnic NXT Compass sensor, EV3 Ultrasonic sensor, Microsoft LifeCam HD-3000, Lego Technics parts

Ubuntu 22.04, ROS2 Humble (RoboStack), BrickPi3

To build [Charlie](Charlie.md)


### Server:

Dell Precision Tower 5810 (48 GB RAM), Nvidia RTX 2070

Ubuntu 18.04, ROS2 Humble (RoboStack), Pytorch 2.1.1 (for object detection)

## Installation

For both robot and server:

Follow the RoboStack installation instructions to install ROS2:

[RoboStack](https://robostack.github.io/GettingStarted.html)

```conda activate ros2```

```mkdir -p ros2_ws/src```

```cd ros2_ws/src```

```git clone https://github.com/jfrancis71/ros2_lego.git```

```cd ..```

```colcon build --symlink-install```


The object detector and person follower requires PyTorch (on the server).
Follow the PyTorch installation instructions with the Conda install:

[PyTorch](https://pytorch.org/get-started/locally/)



## Setup

```cd ros2_ws```

```source ./install/setup.bash```

Run on the robot:

```ros2 launch lego charlie_launch.py```


Run on a controlling PC:

To control by keyboard:

```ros2 run teleop_twist_keyboard teleop_twist_keyboard```

To control by joystick:

```ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox'```
