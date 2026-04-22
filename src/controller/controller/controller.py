#!/usr/bin/env python3

import json
import os
import sys
import time
import uuid

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


MANAGER_CONTROL_TOPIC = "/isaacsim_manager/control"
MANAGER_ACK_TOPIC = "/isaacsim_manager/ack"
MANAGER_WAIT_TIMEOUT_SECONDS = 10.0
PUBLISHER_MATCH_TIMEOUT_SECONDS = 5.0
ACK_TIMEOUT_SECONDS = 180.0


def _topic_exists(node: Node, topic_name: str) -> bool:
    return any(
        name == topic_name for name, _types in node.get_topic_names_and_types()
    )


def main() -> None:
    rclpy.init()
    node = Node("controller")
    manager_ack: dict[str, object] | None = None

    def handle_ack(message: String) -> None:
        nonlocal manager_ack
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            node.get_logger().warning(f"Ignoring invalid manager ack: {message.data!r}")
            return
        if payload.get("id") == request_id:
            manager_ack = payload

    node.declare_parameter("headless", False)
    node.declare_parameter("configuration", "")
    node.declare_parameter("command", "")
    headless = node.get_parameter("headless").value
    configuration = str(node.get_parameter("configuration").value).strip()
    command_param = str(node.get_parameter("command").value).strip()
    node.get_logger().info("rclpy is available")
    node.get_logger().info(
        f"RMW_IMPLEMENTATION={os.environ.get('RMW_IMPLEMENTATION', 'unset')}"
    )
    node.get_logger().info(f"headless={headless}")
    node.get_logger().info(f"configuration={configuration}")
    node.get_logger().info(f"command={command_param}")

    request_id = uuid.uuid4().hex
    command_name = command_param or ("start_headless" if headless else "start")
    command = String()
    command.data = json.dumps(
        {
            "id": request_id,
            "command": command_name,
            "configuration": configuration,
        },
        separators=(",", ":"),
    )
    ack_subscription = node.create_subscription(String, MANAGER_ACK_TOPIC, handle_ack, 10)
    node.get_logger().info(
        f"Waiting up to {MANAGER_WAIT_TIMEOUT_SECONDS:.0f}s for "
        f"Isaac Sim manager topic {MANAGER_CONTROL_TOPIC!r}"
    )
    node.get_logger().info(
        f"Will send manager command id={request_id!r} command={command_name!r} "
        f"on {MANAGER_CONTROL_TOPIC!r}"
    )

    deadline = time.monotonic() + MANAGER_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if _topic_exists(node, MANAGER_CONTROL_TOPIC):
            break
    else:
        node.get_logger().fatal(
            f"Timed out waiting for manager topic: {MANAGER_CONTROL_TOPIC}"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    publisher = node.create_publisher(String, MANAGER_CONTROL_TOPIC, 10)
    # Keep the subscription alive for the request/ack exchange.
    _ = ack_subscription
    node.get_logger().info(
        f"Waiting up to {PUBLISHER_MATCH_TIMEOUT_SECONDS:.0f}s for a manager "
        "subscriber to match"
    )
    deadline = time.monotonic() + PUBLISHER_MATCH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if publisher.get_subscription_count() > 0:
            break
    else:
        node.get_logger().fatal(
            f"No subscribers matched for manager topic: {MANAGER_CONTROL_TOPIC}"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    publisher.publish(command)
    node.get_logger().info(
        f"Published manager command id={request_id!r} command={command_name!r} "
        f"to {MANAGER_CONTROL_TOPIC!r}"
    )

    deadline = time.monotonic() + ACK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if manager_ack is not None:
            break
    else:
        node.get_logger().fatal(
            f"Timed out waiting for manager ack id={request_id!r} "
            f"on {MANAGER_ACK_TOPIC!r}"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    success = bool(manager_ack.get("success"))
    ack_message = str(manager_ack.get("message", ""))
    if success:
        node.get_logger().info(
            f"Manager confirmed command id={request_id!r}: {ack_message}"
        )
    else:
        node.get_logger().fatal(
            f"Manager rejected command id={request_id!r}: {ack_message}"
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info("Controller is running")

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
