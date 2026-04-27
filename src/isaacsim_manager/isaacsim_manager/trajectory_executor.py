import threading
import time
from copy import deepcopy

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


DEFAULT_ACTION_NAME = "right_arm_controller/follow_joint_trajectory"
DEFAULT_JOINT_COMMAND_TOPIC = "/joint_command"
DEFAULT_JOINT_STATE_TOPIC = "/isaac_joint_states"
DEFAULT_GOAL_POSITION_TOLERANCE = 0.15
DEFAULT_STOPPED_VELOCITY_TOLERANCE = 0.2
DEFAULT_GOAL_TIME_TOLERANCE_SECONDS = 4.0
COMMAND_RATE_HZ = 60.0
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


def _duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def _point_time(point) -> float:
    return _duration_seconds(point.time_from_start)


def _result(error_code: int, error_string: str = ""):
    result = FollowJointTrajectory.Result()
    result.error_code = error_code
    result.error_string = error_string
    return result


class IsaacSimTrajectoryExecutor:
    def __init__(
        self,
        *,
        action_name: str = DEFAULT_ACTION_NAME,
        joint_command_topic: str = DEFAULT_JOINT_COMMAND_TOPIC,
        joint_state_topic: str = DEFAULT_JOINT_STATE_TOPIC,
    ) -> None:
        self._action_name = action_name
        self._joint_command_topic = joint_command_topic
        self._joint_state_topic = joint_state_topic
        self._latest_joint_state: JointState | None = None
        self._latest_joint_positions: dict[str, float] = {}
        self._latest_joint_velocities: dict[str, float] = {}
        self._state_lock = threading.Lock()

        self._node = Node("isaacsim_follow_joint_trajectory")
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._joint_command_publisher = self._node.create_publisher(
            JointState,
            joint_command_topic,
            10,
        )
        self._joint_state_subscription = self._node.create_subscription(
            JointState,
            joint_state_topic,
            self._handle_joint_state,
            10,
        )
        self._action_server = ActionServer(
            self._node,
            FollowJointTrajectory,
            action_name,
            execute_callback=self._execute_goal,
            goal_callback=self._handle_goal_request,
            cancel_callback=self._handle_cancel_request,
        )
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="isaacsim_follow_joint_trajectory_executor",
            daemon=True,
        )
        self._spin_thread.start()
        self._node.get_logger().info(
            "Isaac Sim trajectory executor ready "
            f"(action={action_name}, command_topic={joint_command_topic}, "
            f"state_topic={joint_state_topic})"
        )

    def _handle_goal_request(self, goal_request) -> GoalResponse:
        trajectory = goal_request.trajectory
        if list(trajectory.joint_names) != list(RIGHT_ARM_JOINTS):
            self._node.get_logger().error(
                "Rejecting trajectory with unexpected joints: "
                + ", ".join(trajectory.joint_names)
            )
            return GoalResponse.REJECT
        if not trajectory.points:
            self._node.get_logger().error("Rejecting empty trajectory.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _handle_cancel_request(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _handle_joint_state(self, message: JointState) -> None:
        positions = {}
        velocities = {}
        for index, joint_name in enumerate(message.name):
            if index < len(message.position):
                positions[joint_name] = float(message.position[index])
            if index < len(message.velocity):
                velocities[joint_name] = float(message.velocity[index])
        with self._state_lock:
            self._latest_joint_state = deepcopy(message)
            self._latest_joint_positions = positions
            self._latest_joint_velocities = velocities

    def _execute_goal(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        points = list(trajectory.points)
        joint_names = list(trajectory.joint_names)
        self._node.get_logger().info(
            f"Executing Isaac Sim trajectory with {len(points)} points."
        )

        start_time = time.monotonic()
        final_time = _point_time(points[-1])
        command_period = 1.0 / COMMAND_RATE_HZ

        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return _result(
                    RESULT_SUCCESSFUL,
                    "Trajectory canceled.",
                )

            elapsed = time.monotonic() - start_time
            command_point = self._interpolated_point(points, elapsed)
            self._publish_command(joint_names, command_point)
            self._publish_feedback(goal_handle, joint_names, command_point)

            if elapsed >= final_time:
                break
            time.sleep(command_period)

        final_point = points[-1]
        deadline = time.monotonic() + self._goal_time_tolerance(goal_handle)
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return _result(
                    RESULT_SUCCESSFUL,
                    "Trajectory canceled.",
                )
            self._publish_command(joint_names, final_point)
            self._publish_feedback(goal_handle, joint_names, final_point)
            if self._goal_reached(joint_names, final_point):
                goal_handle.succeed()
                self._node.get_logger().info("Isaac Sim trajectory goal reached.")
                return _result(RESULT_SUCCESSFUL)
            time.sleep(command_period)

        error = self._goal_error_summary(joint_names, final_point)
        goal_handle.abort()
        self._node.get_logger().warning(f"Isaac Sim trajectory did not settle: {error}")
        return _result(RESULT_GOAL_TOLERANCE_VIOLATED, error)

    def _goal_time_tolerance(self, goal_handle) -> float:
        tolerance = _duration_seconds(goal_handle.request.goal_time_tolerance)
        if tolerance > 0.0:
            return tolerance
        return DEFAULT_GOAL_TIME_TOLERANCE_SECONDS

    def _interpolated_point(self, points, elapsed: float):
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

    def _publish_command(self, joint_names: list[str], point) -> None:
        message = JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.name = joint_names
        message.position = list(point.positions)
        if point.velocities:
            message.velocity = list(point.velocities)
        self._joint_command_publisher.publish(message)

    def _publish_feedback(self, goal_handle, joint_names: list[str], desired) -> None:
        actual = self._actual_point(joint_names)
        if actual is None:
            return
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = joint_names
        feedback.desired = desired
        feedback.actual = actual
        feedback.error.positions = [
            actual.positions[index] - desired.positions[index]
            for index in range(len(joint_names))
        ]
        if desired.velocities and actual.velocities:
            feedback.error.velocities = [
                actual.velocities[index] - desired.velocities[index]
                for index in range(len(joint_names))
            ]
        goal_handle.publish_feedback(feedback)

    def _actual_point(self, joint_names: list[str]):
        with self._state_lock:
            positions = [self._latest_joint_positions.get(name) for name in joint_names]
            velocities = [self._latest_joint_velocities.get(name, 0.0) for name in joint_names]
        if any(position is None for position in positions):
            return None
        point = JointTrajectoryPoint()
        point.positions = [float(position) for position in positions]
        point.velocities = [float(velocity) for velocity in velocities]
        return point

    def _goal_reached(self, joint_names: list[str], final_point) -> bool:
        actual = self._actual_point(joint_names)
        if actual is None:
            return False
        for index in range(len(joint_names)):
            position_error = abs(actual.positions[index] - final_point.positions[index])
            velocity = abs(actual.velocities[index]) if actual.velocities else 0.0
            if position_error > DEFAULT_GOAL_POSITION_TOLERANCE:
                return False
            if velocity > DEFAULT_STOPPED_VELOCITY_TOLERANCE:
                return False
        return True

    def _goal_error_summary(self, joint_names: list[str], final_point) -> str:
        actual = self._actual_point(joint_names)
        if actual is None:
            return "no complete joint feedback"
        errors = []
        for index, joint_name in enumerate(joint_names):
            errors.append(
                f"{joint_name}: position_error="
                f"{actual.positions[index] - final_point.positions[index]:.4f}, "
                f"velocity={actual.velocities[index]:.4f}"
            )
        return "; ".join(errors)

    def close(self) -> None:
        self._action_server.destroy()
        self._executor.remove_node(self._node)
        self._executor.shutdown()
        self._node.destroy_node()
