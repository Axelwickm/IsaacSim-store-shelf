#!/usr/bin/env python3

from typing import Any

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String

from .coordination import decode_payload, encode_payload
PICK_STEP_SEQUENCE = (
    "approach_pregrasp",
    "move_to_grasp",
    "close_gripper",
    "retract",
    "move_to_drop",
    "open_gripper",
)


class MotionCoordinatorNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_coordinator")
        self.declare_parameter("arm_state_topic", "/motion/arm_state")
        self.declare_parameter("coordinator_state_topic", "/motion/coordinator_state")
        self.declare_parameter("selected_item_point_topic", "/vision/selected_item_point")
        self.declare_parameter(
            "selected_item_moveit_point_topic",
            "/vision/selected_item_moveit_point",
        )
        self.declare_parameter("latched_target_point_topic", "/motion/latched_target_point")

        self._arm_state_topic = str(self.get_parameter("arm_state_topic").value)
        self._coordinator_state_topic = str(
            self.get_parameter("coordinator_state_topic").value
        )
        self._selected_item_point_topic = str(
            self.get_parameter("selected_item_point_topic").value
        )
        self._selected_item_moveit_point_topic = str(
            self.get_parameter("selected_item_moveit_point_topic").value
        )
        self._latched_target_point_topic = str(
            self.get_parameter("latched_target_point_topic").value
        )
        self._arm_states: dict[str, dict[str, Any]] = {}
        self._last_assignment_arm: str | None = None
        self._assigned_target: dict[str, Any] | None = None
        self._active_step: dict[str, Any] | None = None
        self._active_step_index = 0
        self._step_command_serial = 0
        self._candidate_points: dict[int, dict[str, dict[str, Any]]] = {}
        self._arm_state_subscription = self.create_subscription(
            String,
            self._arm_state_topic,
            self._handle_arm_state,
            10,
        )
        self._selected_item_point_subscription = self.create_subscription(
            PointStamped,
            self._selected_item_point_topic,
            self._handle_selected_item_point,
            10,
        )
        self._selected_item_moveit_point_subscription = self.create_subscription(
            PointStamped,
            self._selected_item_moveit_point_topic,
            self._handle_selected_item_moveit_point,
            10,
        )
        self._coordinator_state_publisher = self.create_publisher(
            String,
            self._coordinator_state_topic,
            10,
        )
        self._latched_target_point_publisher = self.create_publisher(
            PointStamped,
            self._latched_target_point_topic,
            10,
        )

        self.get_logger().info(
            f"Motion coordinator ready (arm_state_topic={self._arm_state_topic}, "
            f"coordinator_state_topic={self._coordinator_state_topic}, "
            f"selected_item_point_topic={self._selected_item_point_topic}, "
            f"selected_item_moveit_point_topic={self._selected_item_moveit_point_topic})"
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
        self._sync_active_step_from_arm_state(arm_side, payload)
        self._maybe_assign_candidate()
        self._maybe_issue_next_step()
        self._publish_state()

    def _handle_selected_item_point(self, message: PointStamped) -> None:
        self._record_candidate_point("world", message)

    def _handle_selected_item_moveit_point(self, message: PointStamped) -> None:
        self._record_candidate_point("moveit", message)

    def _record_candidate_point(self, point_type: str, message: PointStamped) -> None:
        stamp_ns = self._message_stamp_ns(message)
        candidate = self._candidate_points.setdefault(stamp_ns, {})
        candidate[point_type] = self._serialize_point(message)
        self._maybe_assign_candidate()
        self._maybe_issue_next_step()
        self._publish_state()

    def _maybe_assign_candidate(self) -> None:
        if self._assigned_target is not None:
            return
        assigned_arm = self._select_assignment_arm()
        if assigned_arm is None:
            return
        for stamp_ns in sorted(self._candidate_points):
            candidate = self._candidate_points[stamp_ns]
            world_point = candidate.get("world")
            moveit_point = candidate.get("moveit")
            if not isinstance(world_point, dict) or not isinstance(moveit_point, dict):
                continue
            if self._point_reserved_by_any_arm(moveit_point):
                continue
            self._assigned_target = {
                "arm_side": assigned_arm,
                "stamp_ns": stamp_ns,
                "world_point": world_point,
                "moveit_point": moveit_point,
            }
            self._active_step = None
            self._active_step_index = 0
            self._last_assignment_arm = assigned_arm
            self._publish_latched_target(world_point)
            self._candidate_points.pop(stamp_ns, None)
            return

    def _point_reserved_by_any_arm(self, point: dict[str, Any]) -> bool:
        for payload in self._arm_states.values():
            if not isinstance(payload, dict):
                continue
            assigned_target = payload.get("assigned_target")
            if not isinstance(assigned_target, dict):
                continue
            assigned_point = assigned_target.get("moveit_point")
            if self._points_match(point, assigned_point):
                return True
        return False

    def _points_match(self, first: dict[str, Any], second: Any) -> bool:
        if not isinstance(second, dict):
            return False
        if str(first.get("frame_id", "")) != str(second.get("frame_id", "")):
            return False
        first_xyz = first.get("xyz") or []
        second_xyz = second.get("xyz") or []
        if len(first_xyz) != 3 or len(second_xyz) != 3:
            return False
        return all(
            abs(float(first_xyz[index]) - float(second_xyz[index])) <= 1e-4
            for index in range(3)
        )

    def _maybe_issue_next_step(self) -> None:
        if self._assigned_target is None or self._active_step is not None:
            return
        assigned_arm = str(self._assigned_target.get("arm_side", "")).strip().lower()
        if not assigned_arm:
            return
        arm_state = self._arm_states.get(assigned_arm, {})
        if not isinstance(arm_state, dict):
            arm_state = {}
        if self._arm_busy(arm_state):
            return
        if self._active_step_index >= len(PICK_STEP_SEQUENCE):
            self._assigned_target = None
            self._active_step_index = 0
            return
        self._step_command_serial += 1
        self._active_step = {
            "arm_side": assigned_arm,
            "command_id": self._step_command_serial,
            "name": PICK_STEP_SEQUENCE[self._active_step_index],
        }

    def _select_assignment_arm(self) -> str | None:
        for payload in self._arm_states.values():
            if not isinstance(payload, dict):
                continue
            if self._arm_busy(payload):
                return None
        available_arms = [
            arm_side
            for arm_side in ("left", "right")
            if self._arm_available_for_assignment(arm_side)
        ]
        if not available_arms:
            return None
        if len(available_arms) == 1:
            return available_arms[0]
        for arm_side in available_arms:
            if arm_side != self._last_assignment_arm:
                return arm_side
        return available_arms[0]

    def _arm_available_for_assignment(self, arm_side: str) -> bool:
        payload = self._arm_states.get(arm_side)
        if not isinstance(payload, dict):
            return True
        if self._arm_busy(payload):
            return False
        return True

    def _arm_busy(self, payload: dict[str, Any]) -> bool:
        if bool(payload.get("motion_active")):
            return True
        step_status = payload.get("step_status")
        if not isinstance(step_status, dict):
            return False
        return str(step_status.get("state", "")).strip().lower() == "running"

    def _sync_active_step_from_arm_state(
        self,
        arm_side: str,
        payload: dict[str, Any],
    ) -> None:
        if self._active_step is None or self._assigned_target is None:
            return
        active_arm = str(self._active_step.get("arm_side", "")).strip().lower()
        if arm_side != active_arm:
            return
        step_status = payload.get("step_status")
        if not isinstance(step_status, dict):
            return
        command_id = int(step_status.get("command_id") or 0)
        expected_id = int(self._active_step.get("command_id") or 0)
        if command_id != expected_id:
            return
        state = str(step_status.get("state", "")).strip().lower()
        if state == "succeeded":
            self._active_step_index += 1
            self._active_step = None
            if self._active_step_index >= len(PICK_STEP_SEQUENCE):
                self._assigned_target = None
                self._active_step_index = 0
            return
        if state == "failed":
            self._active_step = None
            self._active_step_index = 0
            self._assigned_target = None

    def _publish_latched_target(self, point: dict[str, Any]) -> None:
        message = self._deserialize_point(point)
        if message is not None:
            self._latched_target_point_publisher.publish(message)

    def _serialize_point(self, message: PointStamped) -> dict[str, Any]:
        return {
            "frame_id": message.header.frame_id,
            "stamp_ns": self._message_stamp_ns(message),
            "xyz": [
                float(message.point.x),
                float(message.point.y),
                float(message.point.z),
            ],
        }

    def _deserialize_point(self, payload: dict[str, Any]) -> PointStamped | None:
        xyz = payload.get("xyz") or []
        if len(xyz) != 3:
            return None
        message = PointStamped()
        message.header.frame_id = str(payload.get("frame_id", ""))
        stamp_ns = int(payload.get("stamp_ns") or 0)
        if stamp_ns > 0:
            message.header.stamp.sec = stamp_ns // 1_000_000_000
            message.header.stamp.nanosec = stamp_ns % 1_000_000_000
        message.point.x = float(xyz[0])
        message.point.y = float(xyz[1])
        message.point.z = float(xyz[2])
        return message

    def _message_stamp_ns(self, message: PointStamped) -> int:
        return (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def _publish_state(self) -> None:
        payload = {
            "active_step": self._active_step,
            "assigned_target": self._assigned_target,
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
