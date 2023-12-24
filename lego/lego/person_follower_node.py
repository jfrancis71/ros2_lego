import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray


class PersonFollowerNode(Node):
    def __init__(self):
        super().__init__("person_follower_node")
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.subscription = self.create_subscription(
            Detection2DArray,
            'detected_objects',
            self.listener_callback, 0)

    def listener_callback(self, msg):
        detection_array = msg.detections
        x = None
        for detection in detection_array:
            if detection.results[0].hypothesis.class_id == "person":
                x = detection.bbox.center.position.x
                size = detection.bbox.size_x
        twist = Twist()
        if x is not None:
            twist.linear.x = -(size-100)/75
            twist.angular.z = -(x-160)/100
        self.publisher.publish(twist)


rclpy.init()
person_follower_node = PersonFollowerNode()
rclpy.spin(person_follower_node)
person_follower_node.destroy_node()
rclpy.shutdown()

