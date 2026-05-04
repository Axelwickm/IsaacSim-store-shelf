#!/usr/bin/env python3

from typing import Any

import rclpy
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
        self.declare_parameter("selected_candidate_topic", "/vision/selected_candidate")
        self.declare_parameter("reset_topic", "/motion/reset")

        self._arm_state_topic = str(self.get_parameter("arm_state_topic").value)
        self._coordinator_state_topic = str(
            self.get_parameter("coordinator_state_topic").value
        )
        self._selected_candidate_topic = str(
            self.get_parameter("selected_candidate_topic").value
        )
        self._reset_topic = str(self.get_parameter("reset_topic").value)
        self._arm_states: dict[str, dict[str, Any]] = {}
        self._last_assignment_arm: str | None = None
        self._assigned_target: dict[str, Any] | None = None
        self._active_step: dict[str, Any] | None = None
        self._active_step_index = 0
        self._step_command_serial = 0
        self._candidate_points: dict[int, dict[str, Any]] = {}
        self._selected_candidate_subscription = self.create_subscription(
            String,
            self._selected_candidate_topic,
            self._handle_selected_candidate,
            10,
        )
        self._arm_state_subscription = self.create_subscription(
            String,
            self._arm_state_topic,
            self._handle_arm_state,
            10,
        )
        self._reset_subscription = self.create_subscription(
            String,
            self._reset_topic,
            self._handle_reset,
            10,
        )
        self._coordinator_state_publisher = self.create_publisher(
            String,
            self._coordinator_state_topic,
            10,
        )

        self.get_logger().info(
            f"Motion coordinator ready (arm_state_topic={self._arm_state_topic}, "
            f"coordinator_state_topic={self._coordinator_state_topic}, "
            f"selected_candidate_topic={self._selected_candidate_topic}, "
            f"reset_topic={self._reset_topic})"
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

    def _handle_selected_candidate(self, message: String) -> None:
        payload = decode_payload(message)
        if payload is None:
            return
        stamp_ns = int(payload.get("stamp_ns") or 0)
        world_point = payload.get("world_point")
        moveit_point = payload.get("moveit_point")
        arm_side = str(payload.get("arm_side", "")).strip().lower()
        if (
            stamp_ns <= 0
            or not isinstance(world_point, dict)
            or not isinstance(moveit_point, dict)
            or arm_side not in {"left", "right"}
        ):
            return
        candidate = {
            "candidate_id": str(payload.get("candidate_id", "")).strip(),
            "stamp_ns": stamp_ns,
            "arm_side": arm_side,
            "query_index": int(payload.get("query_index") or -1),
            "selection_mode": str(payload.get("selection_mode", "")).strip(),
            "selected_value": float(payload.get("selected_value") or 0.0),
            "value_left": float(payload.get("value_left") or 0.0),
            "value_right": float(payload.get("value_right") or 0.0),
            "world_point": world_point,
            "moveit_point": moveit_point,
        }
        self._candidate_points[stamp_ns] = candidate
        self._maybe_assign_candidate()
        self._maybe_issue_next_step()
        self._publish_state()

    def _handle_reset(self, message: String) -> None:
        payload = decode_payload(message)
        reset_reason = ""
        if isinstance(payload, dict):
            reset_reason = str(payload.get("reason", "")).strip()
        self._arm_states = {}
        self._assigned_target = None
        self._active_step = None
        self._active_step_index = 0
        self._candidate_points.clear()
        self.get_logger().info(
            "Received motion reset"
            + (f" reason={reset_reason}" if reset_reason else "")
        )
        self._publish_state()

    def _maybe_assign_candidate(self) -> None:
        if self._assigned_target is not None:
            return
        for stamp_ns in sorted(self._candidate_points):
            candidate = self._candidate_points[stamp_ns]
            world_point = candidate.get("world_point")
            moveit_point = candidate.get("moveit_point")
            if not isinstance(world_point, dict) or not isinstance(moveit_point, dict):
                continue
            requested_arm = str(candidate.get("arm_side", "")).strip().lower()
            if requested_arm:
                if not self._arm_available_for_assignment(requested_arm):
                    continue
                assigned_arm = requested_arm
            else:
                assigned_arm = self._select_assignment_arm()
                if assigned_arm is None:
                    return
            if self._point_reserved_by_any_arm(moveit_point):
                continue
            self._assigned_target = {
                "arm_side": assigned_arm,
                "stamp_ns": stamp_ns,
                "candidate_id": str(candidate.get("candidate_id", "")).strip(),
                "query_index": int(candidate.get("query_index") or -1),
                "selection_mode": str(candidate.get("selection_mode", "")).strip(),
                "selected_value": float(candidate.get("selected_value") or 0.0),
                "value_left": float(candidate.get("value_left") or 0.0),
                "value_right": float(candidate.get("value_right") or 0.0),
                "world_point": world_point,
                "moveit_point": moveit_point,
            }
            self._active_step = None
            self._active_step_index = 0
            self._last_assignment_arm = assigned_arm
            self.get_logger().info(
                "Assigned candidate "
                f"{self._assigned_target['candidate_id'] or stamp_ns} "
                f"to arm={assigned_arm} "
                f"moveit={self._point_xyz_text(moveit_point)}"
            )
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

    def _point_xyz_text(self, point: dict[str, Any]) -> str:
        xyz = point.get("xyz") or []
        if len(xyz) != 3:
            return "unknown"
        return f"({float(xyz[0]):.3f}, {float(xyz[1]):.3f}, {float(xyz[2]):.3f})"

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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
