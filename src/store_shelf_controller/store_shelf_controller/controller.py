#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node


def main() -> None:
    print("hello world")
    print(f"ROS_DISTRO={os.environ.get('ROS_DISTRO', 'unset')}")

    rclpy.init()
    node = Node("controller")
    node.get_logger().info("rclpy is available")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
