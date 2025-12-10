#!/usr/bin/env python3
"""
Manual /cmd_vel publisher to verify TurtleBot3 motion.
- Publishes forward, backward, left turn, right turn in sequence.
- Use ROS 2 environment (source setup.bash) before running.
"""

import argparse
import time
from typing import Iterable, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def publish_for(node: Node, pub, duration: float, linear_x: float, angular_z: float, rate_hz: float = 10.0):
    """Publish a constant Twist for a given duration at the specified rate."""
    twist = Twist()
    twist.linear.x = linear_x
    twist.angular.z = angular_z

    period = 1.0 / rate_hz
    end_time = time.monotonic() + duration
    while rclpy.ok() and time.monotonic() < end_time:
        pub.publish(twist)
        node.get_logger().info(f"cmd_vel: lin={linear_x:.2f} ang={angular_z:.2f}")
        time.sleep(period)


def run_sequence(node: Node, pub, tests: Iterable[Tuple[str, float, float]], duration: float, pause: float):
    """Run a list of (label, linear_x, angular_z) commands sequentially."""
    stop = Twist()
    for label, lin, ang in tests:
        node.get_logger().info(f"=== {label} for {duration}s ===")
        publish_for(node, pub, duration, lin, ang)
        pub.publish(stop)
        time.sleep(pause)
    node.get_logger().info("Sequence complete.")


def main():
    parser = argparse.ArgumentParser(description="Manual /cmd_vel test (forward/backward/turn).")
    parser.add_argument("--speed", type=float, default=0.15, help="Linear speed magnitude for forward/backward (m/s).")
    parser.add_argument("--turn", type=float, default=0.30, help="Angular speed magnitude for turns (rad/s).")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration per command (seconds).")
    parser.add_argument("--pause", type=float, default=1.0, help="Pause between commands (seconds).")
    args = parser.parse_args()

    rclpy.init()
    node = Node("manual_cmd_vel_test")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)

    tests = [
        ("FORWARD", +args.speed, 0.0),
        ("BACKWARD", -args.speed, 0.0),
        ("TURN_LEFT", 0.0, +args.turn),
        ("TURN_RIGHT", 0.0, -args.turn),
    ]

    try:
        run_sequence(node, pub, tests, duration=args.duration, pause=args.pause)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted. Sending stop.")
        pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
