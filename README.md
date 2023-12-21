# thomas
Thomas is a collection of ROS2 packages for a lego Technic robot (Thomas) running ROS2

```conda activate ros2```

```cd ros2_ws```

```colcon build --symlink-install```

```source ./install/setup.bash```

```ros2 launch thomas robot_launch.py```


To control by keyboard:

```ros2 run teleop_twist_keyboard teleop_twist_keyboard```
