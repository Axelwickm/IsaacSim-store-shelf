#!/usr/bin/env python3

from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .coordination import decode_payload, encode_payload


class MotionCoordinatorNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_coordinator")
        self.declare_parameter("arm_state_topic", "/motion/arm_state")
        self.declare_parameter("coordinator_state_topic", "/motion/coordinator_state")
        self.declare_parameter("state_stale_seconds", 5.0)

        self._arm_state_topic = str(self.get_parameter("arm_state_topic").value)
        self._coordinator_state_topic = str(
            self.get_parameter("coordinator_state_topic").value
        )
        self._state_stale_seconds = max(
            float(self.get_parameter("state_stale_seconds").value),
            0.5,
        )
        self._arm_states: dict[str, dict[str, Any]] = {}
        self._planning_owner: str | None = None
        self._arm_state_subscription = self.create_subscription(
            String,
            self._arm_state_topic,
            self._handle_arm_state,
            10,
        )
        self._coordinator_state_publisher = self.create_publisher(
            String,
            self._coordinator_state_topic,
            10,
        )
        self._stale_state_timer = self.create_timer(1.0, self._prune_stale_states)

        self.get_logger().info(
            f"Motion coordinator ready (arm_state_topic={self._arm_state_topic}, "
            f"coordinator_state_topic={self._coordinator_state_topic})"
        )

    def _handle_arm_state(self, message: String) -> None:
        payload = decode_payload(message)
        if payload is None:
            self.get_logger().warning(
                "Ignoring invalid arm coordination state payload.",
                throttle_duration_sec=5.0,
            )
            return
        arm_side = str(payload.get("arm_side", "")).strip().lower()
        if not arm_side:
            return
        self._arm_states[arm_side] = payload
        self._planning_owner = self._select_planning_owner()
        self._publish_state()

    def _prune_stale_states(self) -> None:
        now_ns = int(self.get_clock().now().nanoseconds)
        stale_sides = []
        for arm_side, payload in self._arm_states.items():
            stamp_ns = int(payload.get("stamp_ns") or 0)
            age_seconds = (now_ns - stamp_ns) * 1e-9 if stamp_ns > 0 else float("inf")
            if age_seconds > self._state_stale_seconds:
                stale_sides.append(arm_side)
        if not stale_sides:
            return
        for arm_side in stale_sides:
            self._arm_states.pop(arm_side, None)
        self._planning_owner = self._select_planning_owner()
        self._publish_state()

    def _select_planning_owner(self) -> str | None:
        requesters: list[tuple[int, str]] = []
        for arm_side, payload in self._arm_states.items():
            if not bool(payload.get("request_planning")):
                continue
            request_stamp_ns = int(payload.get("request_stamp_ns") or payload.get("stamp_ns") or 0)
            requesters.append((request_stamp_ns, arm_side))
        if not requesters:
            return None
        requesters.sort(key=lambda item: (item[0], item[1]))
        if self._planning_owner is not None:
            for _stamp_ns, arm_side in requesters:
                if arm_side == self._planning_owner:
                    return self._planning_owner
        return requesters[0][1]

    def _publish_state(self) -> None:
        payload = {
            "planning_owner": self._planning_owner,
            "arms": self._arm_states,
            "stamp_ns": int(self.get_clock().now().nanoseconds),
        }
        self._coordinator_state_publisher.publish(encode_payload(payload))


def main() -> None:
    rclpy.init()
    node = MotionCoordinatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
