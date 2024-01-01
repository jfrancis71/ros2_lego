# lego
Lego is a ROS2 package for a lego Technic robot (Thomas) running ROS2

![Henry2](https://drive.google.com/uc?id=1GdqDXQZsIsTLFUqVw2gdJ1S9lr9-x4DP&export=download)

## Components

### Robot:

Raspberry Pi 3 Model B+, Dexter Industries BrickPi3, EV3 Motors, HiTechnic NXT Compass sensor, EV3 Ultrasonic sensor, Microsoft LifeCam HD-3000, Lego Technics parts

Ubuntu 22.04, ROS2 Humble (RoboStack), BrickPi3

### Server:

Dell Precision Tower 5810 (48 GB RAM), Nvidia RTX 2070

Ubuntu 18.04, ROS2 Humble (RoboStack), Pytorch 2.1.1 (for object dection)



## Setup

```conda activate ros2```

```cd ros2_ws```

```colcon build --symlink-install```

```source ./install/setup.bash```

Run on the robot:

```ros2 launch lego thomas_launch.py```


Run on a controlling PC:

To control by keyboard:

```ros2 run teleop_twist_keyboard teleop_twist_keyboard```

To control by joystick:

```ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox'```
