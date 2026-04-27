#!/usr/bin/env python3

import time
from threading import Lock

import yaml
from ament_index_python.packages import get_package_share_directory

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointCommandTrajectoryController(Node):
    def __init__(self) -> None:
        super().__init__("joint_command_trajectory_controller")
        self.declare_parameter("action_name", "fake_right_arm_controller/follow_joint_trajectory")
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("status_rate_hz", 60.0)
        self.declare_parameter("goal_tolerance", 0.01)
        self.declare_parameter("goal_settle_timeout", 2.0)
        self.declare_parameter("joint_limit_margin", 0.0)

        action_name = str(self.get_parameter("action_name").value)
        joint_command_topic = str(self.get_parameter("joint_command_topic").value)
        joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._status_period = 1.0 / max(
            float(self.get_parameter("status_rate_hz").value),
            1.0,
        )
        self._goal_tolerance = max(float(self.get_parameter("goal_tolerance").value), 0.0)
        self._goal_settle_timeout = max(
            float(self.get_parameter("goal_settle_timeout").value),
            0.0,
        )
        self._joint_limit_margin = max(
            float(self.get_parameter("joint_limit_margin").value),
            0.0,
        )
        self._joint_limits = self._load_joint_limits()
        self._latest_joint_positions: dict[str, float] = {}
        self._joint_state_lock = Lock()
        self._joint_command_publisher = self.create_publisher(
            JointState,
            joint_command_topic,
            10,
        )
        self._joint_state_subscription = self.create_subscription(
            JointState,
            joint_state_topic,
            self._handle_joint_state,
            10,
        )
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            action_name,
            execute_callback=self._execute_goal,
            goal_callback=self._handle_goal,
            cancel_callback=self._handle_cancel,
        )
        self.get_logger().info(
            f"Serving FollowJointTrajectory on {action_name} and publishing Isaac joint "
            f"commands to {joint_command_topic}"
        )

    def _handle_joint_state(self, message: JointState) -> None:
        if not message.name or not message.position:
            return
        with self._joint_state_lock:
            for name, position in zip(message.name, message.position):
                self._latest_joint_positions[name] = position

    def _load_joint_limits(self) -> dict[str, tuple[float, float]]:
        package_dir = get_package_share_directory("yumi_moveit_config")
        config_path = f"{package_dir}/config/joint_limits.yaml"
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        limits: dict[str, tuple[float, float]] = {}
        for joint_name, config in payload.get("joint_limits", {}).items():
            min_position = config.get("min_position")
            max_position = config.get("max_position")
            if min_position is None or max_position is None:
                continue
            limits[joint_name] = (float(min_position), float(max_position))
        return limits

    def _clamp_positions(self, joint_names: list[str], positions: list[float]) -> list[float]:
        clamped: list[float] = []
        for joint_name, position in zip(joint_names, positions):
            limits = self._joint_limits.get(joint_name)
            if limits is None:
                clamped.append(position)
                continue
            minimum, maximum = limits
            minimum += self._joint_limit_margin
            maximum -= self._joint_limit_margin
            clamped.append(min(max(position, minimum), maximum))
        return clamped

    def _joint_state_snapshot(self, joint_names: list[str]) -> list[float | None]:
        with self._joint_state_lock:
            return [self._latest_joint_positions.get(joint_name) for joint_name in joint_names]

    def _max_abs_error(
        self,
        actual_positions: list[float | None],
        target_positions: list[float],
    ) -> float | None:
        errors = [
            abs(actual - target)
            for actual, target in zip(actual_positions, target_positions)
            if actual is not None
        ]
        if not errors:
            return None
        return max(errors)

    def _format_joint_positions(
        self,
        joint_names: list[str],
        positions: list[float | None],
    ) -> str:
        return ", ".join(
            f"{joint_name}={'?' if position is None else f'{position:.4f}'}"
            for joint_name, position in zip(joint_names, positions)
        )

    def _outside_limits(self, joint_name: str, position: float) -> bool:
        limits = self._joint_limits.get(joint_name)
        if limits is None:
            return False
        minimum, maximum = limits
        return position < minimum or position > maximum

    def _is_goal_reached(self, joint_names: list[str], positions: list[float]) -> bool:
        with self._joint_state_lock:
            for joint_name, target in zip(joint_names, positions):
                actual = self._latest_joint_positions.get(joint_name)
                if actual is None:
                    return False
                if self._outside_limits(joint_name, actual):
                    return False
                if abs(actual - target) > self._goal_tolerance:
                    return False
        return True

    def _publish_joint_command(
        self,
        joint_names: list[str],
        positions: list[float],
    ) -> None:
        joint_command = JointState()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.name = joint_names
        joint_command.position = positions
        self._joint_command_publisher.publish(joint_command)

    def _handle_goal(self, goal_request: FollowJointTrajectory.Goal) -> GoalResponse:
        if not goal_request.trajectory.joint_names:
            self.get_logger().warning("Rejected trajectory goal with no joint names.")
            return GoalResponse.REJECT
        if not goal_request.trajectory.points:
            self.get_logger().warning("Rejected trajectory goal with no trajectory points.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _handle_cancel(self, _goal_handle) -> CancelResponse:
        self.get_logger().info("Accepted trajectory cancel request.")
        return CancelResponse.ACCEPT

    def _execute_goal(self, goal_handle) -> FollowJointTrajectory.Result:
        trajectory = goal_handle.request.trajectory
        joint_names = list(trajectory.joint_names)
        start_time = time.monotonic()

        self.get_logger().info(
            f"Executing trajectory with {len(trajectory.points)} point(s) for joints: "
            + ", ".join(joint_names)
        )
        start_positions = self._joint_state_snapshot(joint_names)
        self.get_logger().info(
            "Trajectory start actual joints: "
            + self._format_joint_positions(joint_names, start_positions)
        )

        final_positions: list[float] | None = None
        unclamped_final_positions: list[float] | None = None
        for index, point in enumerate(trajectory.points):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "Trajectory goal canceled during execution."
                return result

            target_time = (
                float(point.time_from_start.sec)
                + float(point.time_from_start.nanosec) / 1_000_000_000.0
            )
            while True:
                remaining = target_time - (time.monotonic() - start_time)
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, self._status_period))

            unclamped_positions = list(point.positions)
            command_positions = self._clamp_positions(joint_names, list(point.positions))
            unclamped_final_positions = unclamped_positions
            final_positions = list(command_positions)

            joint_command = JointState()
            joint_command.header.stamp = self.get_clock().now().to_msg()
            joint_command.name = joint_names
            joint_command.position = command_positions
            if point.velocities:
                joint_command.velocity = list(point.velocities)
            if point.effort:
                joint_command.effort = list(point.effort)
            self._joint_command_publisher.publish(joint_command)

            feedback = FollowJointTrajectory.Feedback()
            feedback.joint_names = joint_names
            feedback.desired = point
            feedback.actual = point
            feedback.error.positions = [0.0] * len(point.positions)
            feedback.error.velocities = [0.0] * len(point.velocities)
            feedback.error.accelerations = [0.0] * len(point.accelerations)
            feedback.error.effort = [0.0] * len(point.effort)
            goal_handle.publish_feedback(feedback)
            self.get_logger().debug(
                f"Published trajectory point {index + 1}/{len(trajectory.points)}."
            )

        if final_positions is not None:
            if unclamped_final_positions is not None and unclamped_final_positions != final_positions:
                self.get_logger().warning(
                    "Clamped final trajectory point from "
                    + self._format_joint_positions(joint_names, unclamped_final_positions)
                    + " to "
                    + self._format_joint_positions(joint_names, final_positions)
                )
            self.get_logger().info(
                "Trajectory final command joints: "
                + self._format_joint_positions(joint_names, final_positions)
            )
            deadline = time.monotonic() + self._goal_settle_timeout
            while not self._is_goal_reached(joint_names, final_positions):
                self._publish_joint_command(joint_names, final_positions)
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result = FollowJointTrajectory.Result()
                    result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                    result.error_string = "Trajectory goal canceled while waiting for settle."
                    return result
                if time.monotonic() >= deadline:
                    actual_positions = self._joint_state_snapshot(joint_names)
                    max_error = self._max_abs_error(actual_positions, final_positions)
                    self.get_logger().error(
                        "Trajectory settle timeout. Actual joints: "
                        + self._format_joint_positions(joint_names, actual_positions)
                        + "; max_error="
                        + ("?" if max_error is None else f"{max_error:.4f}")
                    )
                    goal_handle.abort()
                    result = FollowJointTrajectory.Result()
                    result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                    result.error_string = "Robot state did not settle to the commanded final point."
                    return result
                time.sleep(self._status_period)
            actual_positions = self._joint_state_snapshot(joint_names)
            max_error = self._max_abs_error(actual_positions, final_positions)
            self.get_logger().info(
                "Trajectory settled. Actual joints: "
                + self._format_joint_positions(joint_names, actual_positions)
                + "; max_error="
                + ("?" if max_error is None else f"{max_error:.4f}")
            )

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = ""
        return result


def main() -> None:
    rclpy.init()
    node = JointCommandTrajectoryController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
