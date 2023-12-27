# lego
Lego is a ROS2 package for a lego Technic robot (Thomas) running ROS2

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
