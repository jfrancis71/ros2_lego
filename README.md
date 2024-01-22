
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
