import threading
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint


DEFAULT_ACTION_NAME = "right_arm_controller/follow_joint_trajectory"
DEFAULT_GOAL_POSITION_TOLERANCE = 0.05
DEFAULT_STOPPED_VELOCITY_TOLERANCE = 0.25
DEFAULT_GOAL_TIME_TOLERANCE_SECONDS = 10.0
MAX_WALL_CLOCK_STEP_SECONDS = 0.25
RESULT_SUCCESSFUL = 0
RESULT_GOAL_TOLERANCE_VIOLATED = -5
RIGHT_ARM_JOINTS = (
    "yumi_joint_1_r",
    "yumi_joint_2_r",
    "yumi_joint_7_r",
    "yumi_joint_3_r",
    "yumi_joint_4_r",
    "yumi_joint_5_r",
    "yumi_joint_6_r",
)
LEFT_ARM_JOINTS = (
    "yumi_joint_1_l",
    "yumi_joint_2_l",
    "yumi_joint_7_l",
    "yumi_joint_3_l",
    "yumi_joint_4_l",
    "yumi_joint_5_l",
    "yumi_joint_6_l",
)
ARM_JOINT_LIMITS_RAD = {
    "yumi_joint_1_l": (-2.84, 2.84),
    "yumi_joint_2_l": (-2.4, 0.65),
    "yumi_joint_7_l": (-2.84, 2.84),
    "yumi_joint_3_l": (-2.0, 1.29),
    "yumi_joint_4_l": (-4.9, 4.9),
    "yumi_joint_5_l": (-1.43, 2.3),
    "yumi_joint_6_l": (-3.89, 3.89),
    "yumi_joint_1_r": (-2.84, 2.84),
    "yumi_joint_2_r": (-2.4, 0.65),
    "yumi_joint_7_r": (-2.84, 2.84),
    "yumi_joint_3_r": (-2.0, 1.29),
    "yumi_joint_4_r": (-4.9, 4.9),
    "yumi_joint_5_r": (-1.43, 2.3),
    "yumi_joint_6_r": (-3.89, 3.89),
}
JOINT_LIMIT_MARGIN_RAD = 1e-3


@dataclass
class ActiveTrajectory:
    goal_handle: object
    joint_names: list[str]
    joint_indices: np.ndarray | None
    points: list[JointTrajectoryPoint]
    final_time: float
    elapsed: float = 0.0
    settle_elapsed: float = 0.0
    last_update_monotonic: float = 0.0
    done_event: threading.Event | None = None
    result: FollowJointTrajectory.Result | None = None


def _duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def _point_time(point: JointTrajectoryPoint) -> float:
    return _duration_seconds(point.time_from_start)


def _result(error_code: int, error_string: str = "") -> FollowJointTrajectory.Result:
    result = FollowJointTrajectory.Result()
    result.error_code = error_code
    result.error_string = error_string
    return result


class IsaacSimTrajectoryExecutor:
    def __init__(
        self,
        *,
        articulation_root_path: str,
        action_name: str = DEFAULT_ACTION_NAME,
        joint_names: tuple[str, ...] = RIGHT_ARM_JOINTS,
    ) -> None:
        self._articulation_root_path = articulation_root_path
        self._action_name = action_name
        self._joint_names = tuple(joint_names)
        self._joint_name_set = set(self._joint_names)
        self._active: ActiveTrajectory | None = None
        self._lock = threading.Lock()
        self._articulation = None
        self._articulation_action_type = None
        self._logged_first_command = False

        node_suffix = action_name.split("/", 1)[0].replace("/", "_")
        self._node = Node(f"isaacsim_follow_joint_trajectory_{node_suffix}")
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._action_server = ActionServer(
            self._node,
            FollowJointTrajectory,
            action_name,
            execute_callback=self._execute_goal,
            cancel_callback=self._handle_cancel_request,
        )
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="isaacsim_follow_joint_trajectory_executor",
            daemon=True,
        )
        self._spin_thread.start()
        self._node.get_logger().info(
            "Isaac Sim direct trajectory executor ready "
            f"(action={action_name}, joints={','.join(self._joint_names)}, articulation={articulation_root_path})"
        )

    def _handle_cancel_request(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_goal(self, goal_handle):
        done_event = threading.Event()
        joint_names = list(goal_handle.request.trajectory.joint_names)
        points = list(goal_handle.request.trajectory.points)
        if set(joint_names) != self._joint_name_set:
            error = (
                "Unexpected trajectory joints: "
                + ", ".join(joint_names)
                + "; expected joint set: "
                + ", ".join(self._joint_names)
            )
            self._node.get_logger().error(error)
            goal_handle.abort()
            return _result(RESULT_GOAL_TOLERANCE_VIOLATED, error)
        if not points:
            error = "Empty trajectory."
            self._node.get_logger().error(error)
            goal_handle.abort()
            return _result(RESULT_GOAL_TOLERANCE_VIOLATED, error)
        with self._lock:
            if self._active is not None:
                error = "A direct Isaac Sim trajectory is already active."
                self._node.get_logger().warning(error)
                goal_handle.abort()
                return _result(RESULT_GOAL_TOLERANCE_VIOLATED, error)
        active = ActiveTrajectory(
            goal_handle=goal_handle,
            joint_names=joint_names,
            joint_indices=None,
            points=points,
            final_time=_point_time(points[-1]),
            last_update_monotonic=time.monotonic(),
            done_event=done_event,
        )
        with self._lock:
            self._active = active
        self._node.get_logger().info(
            "Executing direct Isaac Sim trajectory "
            f"with {len(points)} points over {active.final_time:.3f}s."
        )
        done_event.wait()
        return active.result or _result(
            RESULT_GOAL_TOLERANCE_VIOLATED,
            "trajectory ended without result",
        )

    def update(self, dt: float) -> None:
        if self._active is None:
            return
        self._ensure_articulation()
        with self._lock:
            active = self._active
        if active is None:
            return
        if active.joint_indices is None:
            active.joint_indices = self._joint_indices(active.joint_names)

        dt = self._trajectory_time_delta(active, dt)
        goal_handle = active.goal_handle
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self._finish(active, RESULT_SUCCESSFUL, "Trajectory canceled.")
            return

        active.elapsed += dt
        if active.elapsed < active.final_time:
            command_point = self._interpolated_point(active.points, active.elapsed)
            self._apply_action(active, command_point)
            self._publish_feedback(active, command_point)
            return

        final_point = active.points[-1]
        self._apply_action(active, final_point)
        self._publish_feedback(active, final_point)
        if self._goal_reached(active, final_point):
            goal_handle.succeed()
            self._finish(active, RESULT_SUCCESSFUL)
            self._node.get_logger().info("Direct Isaac Sim trajectory goal reached.")
            return

        active.settle_elapsed += dt
        if active.settle_elapsed <= self._goal_time_tolerance(goal_handle):
            return

        error = self._goal_error_summary(active, final_point)
        goal_handle.abort()
        self._finish(active, RESULT_GOAL_TOLERANCE_VIOLATED, error)
        self._node.get_logger().warning(f"Direct Isaac Sim trajectory did not settle: {error}")

    def has_active_trajectory(self) -> bool:
        with self._lock:
            return self._active is not None

    def _trajectory_time_delta(self, active: ActiveTrajectory, fallback_dt: float) -> float:
        now = time.monotonic()
        if active.last_update_monotonic <= 0.0:
            active.last_update_monotonic = now
            return fallback_dt
        elapsed = now - active.last_update_monotonic
        active.last_update_monotonic = now
        if elapsed <= 0.0:
            return fallback_dt
        return min(elapsed, MAX_WALL_CLOCK_STEP_SECONDS)

    def _finish(self, active: ActiveTrajectory, error_code: int, error_string: str = "") -> None:
        active.result = _result(error_code, error_string)
        with self._lock:
            if self._active is active:
                self._active = None
        if active.done_event is not None:
            active.done_event.set()

    def _ensure_articulation(self) -> None:
        if self._articulation is not None:
            return
        try:
            from isaacsim.core.prims import SingleArticulation
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.articulations import Articulation as SingleArticulation
            from omni.isaac.core.utils.types import ArticulationAction

        self._articulation_action_type = ArticulationAction
        self._articulation = SingleArticulation(
            prim_path=self._articulation_root_path,
            name=f"yumi_direct_trajectory_articulation_{self._action_name.split('/', 1)[0]}",
        )
        self._articulation.initialize()
        self._node.get_logger().info(
            "Initialized direct Isaac articulation controller for "
            f"{self._articulation_root_path}"
        )

    def _joint_indices(self, joint_names: list[str]) -> np.ndarray:
        return np.array(
            [self._articulation.get_dof_index(joint_name) for joint_name in joint_names],
            dtype=np.int32,
        )

    def _goal_time_tolerance(self, goal_handle) -> float:
        tolerance = _duration_seconds(goal_handle.request.goal_time_tolerance)
        if tolerance > 0.0:
            return tolerance
        return DEFAULT_GOAL_TIME_TOLERANCE_SECONDS

    def _interpolated_point(
        self,
        points: list[JointTrajectoryPoint],
        elapsed: float,
    ) -> JointTrajectoryPoint:
        if elapsed <= _point_time(points[0]):
            return points[0]
        for next_index in range(1, len(points)):
            previous = points[next_index - 1]
            current = points[next_index]
            previous_time = _point_time(previous)
            current_time = _point_time(current)
            if elapsed > current_time:
                continue
            span = max(current_time - previous_time, 1e-9)
            ratio = min(max((elapsed - previous_time) / span, 0.0), 1.0)
            point = deepcopy(current)
            point.positions = [
                previous.positions[index]
                + (current.positions[index] - previous.positions[index]) * ratio
                for index in range(len(current.positions))
            ]
            if previous.velocities and current.velocities:
                point.velocities = [
                    previous.velocities[index]
                    + (current.velocities[index] - previous.velocities[index]) * ratio
                    for index in range(len(current.velocities))
                ]
            return point
        return points[-1]

    def _apply_action(self, active: ActiveTrajectory, point: JointTrajectoryPoint) -> None:
        positions = np.array(
            self._clamped_positions(active.joint_names, point.positions),
            dtype=np.float64,
        )
        action = self._articulation_action_type(
            joint_positions=positions,
            joint_indices=active.joint_indices,
        )
        self._articulation.apply_action(action)
        self._articulation.set_joint_positions(
            positions,
            joint_indices=active.joint_indices,
        )
        if not self._logged_first_command:
            self._logged_first_command = True
            self._node.get_logger().info(
                "Applied first direct Isaac joint command: "
                + ", ".join(
                    f"{name}={position:.4f}"
                    for name, position in zip(active.joint_names, positions, strict=False)
                )
            )

    def _clamped_positions(
        self,
        joint_names: list[str],
        positions: list[float],
    ) -> list[float]:
        clamped = []
        for joint_name, position in zip(joint_names, positions):
            if joint_name not in ARM_JOINT_LIMITS_RAD:
                raise KeyError(f"Missing joint limits for {joint_name}")
            lower, upper = ARM_JOINT_LIMITS_RAD[joint_name]
            clamped.append(
                min(
                    max(position, lower + JOINT_LIMIT_MARGIN_RAD),
                    upper - JOINT_LIMIT_MARGIN_RAD,
                )
            )
        return clamped

    def _publish_feedback(
        self,
        active: ActiveTrajectory,
        desired: JointTrajectoryPoint,
    ) -> None:
        actual = self._actual_point(active)
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = active.joint_names
        feedback.desired = desired
        feedback.actual = actual
        feedback.error.positions = [
            actual.positions[index] - desired.positions[index]
            for index in range(len(active.joint_names))
        ]
        if desired.velocities and actual.velocities:
            feedback.error.velocities = [
                actual.velocities[index] - desired.velocities[index]
                for index in range(len(active.joint_names))
            ]
        active.goal_handle.publish_feedback(feedback)

    def _actual_point(self, active: ActiveTrajectory) -> JointTrajectoryPoint:
        point = JointTrajectoryPoint()
        point.positions = list(
            self._articulation.get_joint_positions(joint_indices=active.joint_indices)
        )
        point.velocities = list(
            self._articulation.get_joint_velocities(joint_indices=active.joint_indices)
        )
        return point

    def _goal_reached(
        self,
        active: ActiveTrajectory,
        final_point: JointTrajectoryPoint,
    ) -> bool:
        actual = self._actual_point(active)
        final_positions = self._clamped_positions(active.joint_names, final_point.positions)
        for index in range(len(active.joint_names)):
            position_error = abs(actual.positions[index] - final_positions[index])
            velocity = abs(actual.velocities[index]) if actual.velocities else 0.0
            if position_error > DEFAULT_GOAL_POSITION_TOLERANCE:
                return False
            if velocity > DEFAULT_STOPPED_VELOCITY_TOLERANCE:
                return False
        return True

    def _goal_error_summary(
        self,
        active: ActiveTrajectory,
        final_point: JointTrajectoryPoint,
    ) -> str:
        actual = self._actual_point(active)
        final_positions = self._clamped_positions(active.joint_names, final_point.positions)
        errors = []
        for index, joint_name in enumerate(active.joint_names):
            errors.append(
                f"{joint_name}: position_error="
                f"{actual.positions[index] - final_positions[index]:.4f}, "
                f"velocity={actual.velocities[index]:.4f}"
            )
        return "; ".join(errors)

    def close(self) -> None:
        self._action_server.destroy()
        self._executor.remove_node(self._node)
        self._executor.shutdown()
        self._node.destroy_node()
