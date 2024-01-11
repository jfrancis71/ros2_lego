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

### Robot

Follow the [RoboStack](https://robostack.github.io/GettingStarted.html) installation instructions to install ROS2

(Ensure you have also followed the step Installation tools for local development in the above instructions)


```
conda activate ros2 (use whatever name here you decided to call this conda environment)
mamba install ros-humble-compressed-image-transport
git clone https://github.com/DexterInd/BrickPi3.git
cd BrickPi3/Software/Python/
pip install .
cd ~
mkdir -p ros2_ws/src
cd ros2_ws/src
git clone https://github.com/jfrancis71/ros2_lego.git
cd ..
colcon build --symlink-install
```

### Server

Follow the [RoboStack](https://robostack.github.io/GettingStarted.html) installation instructions to install ROS2

(Ensure you have also followed the step Installation tools for local development in the above instructions)


```
conda activate ros2 (use whatever name here you decided to call this conda environment)
mamba install ros-humble-compressed-image-transport
mamba install ros-humble-vision-msgs
mkdir -p ros2_ws/src
cd ros2_ws/src
git clone https://github.com/jfrancis71/ros2_lego.git
cd ..
colcon build --symlink-install
```


The object detector and person follower requires PyTorch (on the server).
Follow the [PyTorch](https://pytorch.org/get-started/locally/) installation instructions with the Conda install (but replace the conda command with mamba)




## Setup Environment

Execute on both robot and server:
```
conda activate ros2 (use whatever name here you decided to call this conda environment)
cd ros2_ws
source ./install/setup.bash
```

Run on the robot:

```ros2 launch charlie robot_launch.py```


## Useful Commands

Run on a controlling PC:

To control by keyboard:

```ros2 run teleop_twist_keyboard teleop_twist_keyboard```

To control by joystick:

Note charlie interprets twist messages in metric, so need to scale joystick. The below config file is based off the teleop_twist_joy/confix/xbox.config.yaml, but modified to scale linear.x
You may need to alter depending on your joystick.

```ros2 launch teleop_twist_joy teleop-launch.py config_filepath:=./src/ros2_lego/lego/resource/xeox_charlie.config.yaml```

To stream from a webcam publishing http mjpeg to ROS2 topic /image (substitute IP address):

```ros2 run image_publisher image_publisher_node http://192.168.1.246/video.mjpg```

installing:
```mamba install ros-humble-image-publisher```
if necessary
