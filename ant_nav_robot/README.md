# Terry (the tri-omniwheeled holonomic robot)

## Overview

Terry is a demonstration tri-omniwheeled robot running ROS2 on a Raspberry Pi. He is built using Lego and EV3 motors with the Dexter Industries BrickPi3+ providing the hardware interface from the Raspberry Pi to the EV3 motors. The terry package demonstrates how to configure the following two packages together to create a working robot:

- A ROS2 omniwheel controller, available from (https://github.com/hijimasa/omni_wheel_controller), which listens to ROS2 twist messages and, using a configuration file specifying robot geometry, issues ROS2 controller motor velocity commands.

- The brickpi3_motors package, available from (https://github.com/jfrancis71/ros2_brickpi3), listens to these motor velocity commands and instructs the BrickPi3 hardware to rotate the motors correspondingly.
(Disclosure: I am the author of the ROS2 BrickPi3 package).

The Omniwheels are not a Lego product, my recollection is that I purchased them from: https://uk.robotshop.com/products/48mm-omniwheel-compatible-servos-lego-mindstorms-nxt

## Lego Assembly

[Lego Assembly Instructions](./lego_assembly/README.md)


## Installation

Follow instructions to build ROS2 BrickPi3 at (https://github.com/jfrancis71/ros2_brickpi3)

In the above step I am recommending Robostack ROS2, but if you are using you're own ROS2 install, you should replace the 'mamba install ros-humble-generate-parameter-library' install in the below instructions with what is appropriate for your ROS2 install.

```
mamba install ros-humble-generate-parameter-library
cd ~/ros2_ws
git -C src clone https://github.com/hijimasa/omni_wheel_controller
git -C src clone https://github.com/jfrancis71/ros2_holonomic_lego.git
colcon build --symlink-install
```

### Troubleshooting
I recommend having a seperate shell running htop so you can monitor progress. The Raspberry Pi 3B+ is quite memory limited which can cause problems installing some packages, particularly the omni_wheel_controller in the last step above. If you see process status 'D' in htop relating to the install processes that persists this can indicate difficulties due to low memory. In this case I suggest before running the colcon build step (the final step in the instructions), adding:

```
export MAKEFLAGS="-j 1" # recommended to reduce memory usage.
```

Also I suggest adding some temporary swap (I found 2GB perfectly sufficient). See discussion from Digital Ocean in the References section. Don't forget to remove the swap after a succesful installation. (A swap file on an SD card will reduce card life significantly)

The succesful install of omni_wheel_controller took about ten minutes.


## Activate Environment

```
mamba activate ros2 # (use the name here you decided to call this conda environment)
cd ~/ros2_ws
source ./install/setup.bash
```

## Verify Install

Activate the motor controller:
```
ros2 launch terry brickpi3_motors_launch.py
```

This should cause the motors to rotate (briefly):
```
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```


## References:

#### Third Party Packages:
- Omni wheel controller: https://github.com/hijimasa/omni_wheel_controller
- ROS2 BrickPi3: https://github.com/jfrancis71/ros2_brickpi3


A Simple Introduction to Omni Roller Robots:
http://modwg.co.uk/wp-content/uploads/2015/06/OmniRoller-Holonomic-Drive-Tutorial.pdf


Northwestern Robotics, Modern Robotics Chapter 13.2
https://www.youtube.com/watch?v=NcOT9hOsceE


Kinematics of Mobile Robots with Omni Directional Wheels:
https://www.youtube.com/watch?v=-wzl8XJopgg&t=1756s


Useful discussion on swap file on Ubuntu:
https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04


Omniwheels:
https://uk.robotshop.com/products/48mm-omniwheel-compatible-servos-lego-mindstorms-nxt


Dexter Industries BrickPi3:
https://www.dexterindustries.com/brickpi-core/

Robot Cheat sheet:
https://www.theroboticsspace.com/assets/article3/ros2_humble_cheat_sheet2.pdf
