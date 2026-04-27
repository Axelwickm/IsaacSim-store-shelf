#!/usr/bin/env python3

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

try:
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        BoundingVolume,
        Constraints,
        MotionPlanRequest,
        OrientationConstraint,
        PlanningOptions,
        PositionConstraint,
        RobotState,
    )
    from rclpy.action import ActionClient
    from shape_msgs.msg import SolidPrimitive

    MOVEIT_ACTIONS_AVAILABLE = True
except ModuleNotFoundError:
    MoveGroup = None
    ActionClient = None
    MOVEIT_ACTIONS_AVAILABLE = False


@dataclass(frozen=True)
class ExternalDependency:
    package_name: str
    purpose: str


@dataclass(frozen=True)
class MotionStep:
    name: str
    pose: PoseStamped | None = None
    gripper_closed: bool | None = None
    force_top_down: bool = False
    top_down_tolerance_xyz: tuple[float, float, float] | None = None
    top_down_weight: float = 1.0


REQUIRED_EXTERNAL_PACKAGES = (
    ExternalDependency("moveit_core", "MoveIt core libraries"),
    ExternalDependency("moveit_ros_move_group", "MoveIt move_group runtime"),
    ExternalDependency("isaac_ros_cumotion", "cuMotion planner runtime"),
)

MOVEIT_ERROR_CODE_NAMES = {
    1: "SUCCESS",
    99999: "FAILURE",
    -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN",
    -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -4: "CONTROL_FAILED",
    -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -6: "TIMED_OUT",
    -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME",
    -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE",
    -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE",
    -24: "SENSOR_INFO_STALE",
    -25: "COMMUNICATION_FAILURE",
    -26: "START_STATE_INVALID",
    -27: "CRASH",
    -28: "ABORT",
    -29: "NO_IK_SOLUTION",
}
class MotionPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_planner")
        self.declare_parameter("planning_group", "yumi_arm")
        self.declare_parameter("right_arm_group", "right_arm")
        self.declare_parameter("end_effector_link", "gripper_r_grasp_frame")
        self.declare_parameter("moveit_target_frame", "yumi_body")
        self.declare_parameter("pipeline_id", "isaac_ros_cumotion")
        self.declare_parameter("planner_id", "cuMotion")
        self.declare_parameter("move_group_action", "/move_action")
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("right_gripper_joint", "gripper_r_joint")
        self.declare_parameter("right_gripper_mimic_joint", "gripper_r_joint_m")
        self.declare_parameter("right_gripper_open_position", 0.025)
        self.declare_parameter("right_gripper_closed_position", 0.0)
        self.declare_parameter("gripper_command_settle_seconds", 1.0)
        self.declare_parameter("target_pose_topic", "/motion/target_pose")
        self.declare_parameter("selected_item_point_topic", "/vision/selected_item_point")
        self.declare_parameter(
            "selected_item_moveit_point_topic",
            "/vision/selected_item_moveit_point",
        )
        self.declare_parameter("latched_target_point_topic", "/motion/latched_target_point")
        self.declare_parameter("clear_latched_target_topic", "/motion/clear_latched_target")
        self.declare_parameter("auto_execute_pick", True)
        self.declare_parameter("plan_only", True)
        self.declare_parameter("pregrasp_offset_xyz", [0.0, 0.0, 0.08])
        self.declare_parameter("grasp_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("retract_offset_xyz", [0.0, 0.0, 0.20])
        self.declare_parameter("drop_position_xyz", [0.25, -0.14, 0.0])
        self.declare_parameter("position_tolerance", 0.025)
        self.declare_parameter("use_top_down_orientation_constraint", True)
        self.declare_parameter("top_down_orientation_xyzw", [0.0, 1.0, 0.0, 0.0])
        self.declare_parameter("top_down_orientation_tolerance_xyz", [0.9, 0.9, 3.14159])
        self.declare_parameter(
            "release_top_down_orientation_tolerance_xyz",
            [0.25, 0.25, 3.14159],
        )
        self.declare_parameter("release_top_down_orientation_weight", 1.0)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("num_planning_attempts", 5)
        self.declare_parameter("max_velocity_scaling_factor", 0.2)
        self.declare_parameter("max_acceleration_scaling_factor", 0.2)

        self._validate_external_dependencies()

        planning_group = str(self.get_parameter("planning_group").value)
        self._right_arm_group = str(self.get_parameter("right_arm_group").value)
        self._end_effector_link = str(self.get_parameter("end_effector_link").value)
        self._moveit_target_frame = str(self.get_parameter("moveit_target_frame").value)
        pipeline_id = str(self.get_parameter("pipeline_id").value)
        planner_id = str(self.get_parameter("planner_id").value)
        move_group_action = str(self.get_parameter("move_group_action").value)
        joint_command_topic = str(self.get_parameter("joint_command_topic").value)
        target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        selected_item_point_topic = str(
            self.get_parameter("selected_item_point_topic").value
        )
        selected_item_moveit_point_topic = str(
            self.get_parameter("selected_item_moveit_point_topic").value
        )
        latched_target_point_topic = str(
            self.get_parameter("latched_target_point_topic").value
        )
        clear_latched_target_topic = str(
            self.get_parameter("clear_latched_target_topic").value
        )
        self._right_gripper_joint = str(self.get_parameter("right_gripper_joint").value)
        self._right_gripper_mimic_joint = str(
            self.get_parameter("right_gripper_mimic_joint").value
        )
        self._right_gripper_open_position = float(
            self.get_parameter("right_gripper_open_position").value
        )
        self._right_gripper_closed_position = float(
            self.get_parameter("right_gripper_closed_position").value
        )
        self._gripper_command_settle_seconds = float(
            self.get_parameter("gripper_command_settle_seconds").value
        )

        self._latched_target_point: PointStamped | None = None
        self._latest_world_selected_item_point: PointStamped | None = None
        self._latest_joint_state: JointState | None = None
        self._joint_state_ready = False
        self._logged_joint_state_names = False
        self._pick_in_progress = False
        self._pick_sequence: list[MotionStep] = []
        self._pick_sequence_index = 0
        self._pick_generation = 0
        self._step_timer: Any | None = None
        self._move_result_timer: Any | None = None
        self._active_goal_handle: Any | None = None
        self._move_group_goal_may_be_active = False
        self._pipeline_id = pipeline_id
        self._planner_id = planner_id
        self._auto_execute_pick = bool(self.get_parameter("auto_execute_pick").value)
        self._plan_only = bool(self.get_parameter("plan_only").value)
        self._pregrasp_offset_xyz = self._float_list_parameter("pregrasp_offset_xyz", 3)
        self._grasp_offset_xyz = self._float_list_parameter("grasp_offset_xyz", 3)
        self._retract_offset_xyz = self._float_list_parameter("retract_offset_xyz", 3)
        self._drop_position_xyz = self._float_list_parameter("drop_position_xyz", 3)
        self._position_tolerance = float(self.get_parameter("position_tolerance").value)
        self._use_top_down_orientation_constraint = bool(
            self.get_parameter("use_top_down_orientation_constraint").value
        )
        self._top_down_orientation_xyzw = self._float_list_parameter(
            "top_down_orientation_xyzw",
            4,
        )
        self._top_down_orientation_tolerance_xyz = self._float_list_parameter(
            "top_down_orientation_tolerance_xyz",
            3,
        )
        self._release_top_down_orientation_tolerance_xyz = tuple(
            self._float_list_parameter(
                "release_top_down_orientation_tolerance_xyz",
                3,
            )
        )
        self._release_top_down_orientation_weight = float(
            self.get_parameter("release_top_down_orientation_weight").value
        )
        self._allowed_planning_time = float(
            self.get_parameter("allowed_planning_time").value
        )
        self._move_group_result_timeout = max(self._allowed_planning_time + 5.0, 10.0)
        self._num_planning_attempts = int(
            self.get_parameter("num_planning_attempts").value
        )
        self._max_velocity_scaling_factor = float(
            self.get_parameter("max_velocity_scaling_factor").value
        )
        self._max_acceleration_scaling_factor = float(
            self.get_parameter("max_acceleration_scaling_factor").value
        )
        self._latched_target_point_publisher = self.create_publisher(
            PointStamped,
            latched_target_point_topic,
            10,
        )
        self._joint_command_publisher = self.create_publisher(
            JointState,
            joint_command_topic,
            10,
        )
        self._pregrasp_pose_publisher = self.create_publisher(
            PoseStamped,
            "/motion/pregrasp_pose",
            10,
        )
        self._grasp_pose_publisher = self.create_publisher(
            PoseStamped,
            "/motion/grasp_pose",
            10,
        )
        self._retract_pose_publisher = self.create_publisher(
            PoseStamped,
            "/motion/retract_pose",
            10,
        )
        self._drop_pose_publisher = self.create_publisher(
            PoseStamped,
            "/motion/drop_pose",
            10,
        )
        self._move_group_client: Any | None = None
        if MOVEIT_ACTIONS_AVAILABLE:
            self._move_group_client = ActionClient(self, MoveGroup, move_group_action)

        self._target_pose_subscription = self.create_subscription(
            PoseStamped,
            target_pose_topic,
            self._handle_target_pose,
            10,
        )
        self._selected_item_point_subscription = self.create_subscription(
            PointStamped,
            selected_item_point_topic,
            self._handle_selected_item_point,
            10,
        )
        self._joint_state_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self._handle_joint_state,
            10,
        )
        self._selected_item_moveit_point_subscription = self.create_subscription(
            PointStamped,
            selected_item_moveit_point_topic,
            self._handle_selected_item_moveit_point,
            10,
        )
        self._clear_latched_target_subscription = self.create_subscription(
            Empty,
            clear_latched_target_topic,
            self._handle_clear_latched_target,
            10,
        )

        self.get_logger().info(
            "Motion planner scaffold ready "
            f"(group={planning_group}, right_arm={self._right_arm_group}, "
            f"target_frame={self._moveit_target_frame}, "
            f"pipeline={pipeline_id}, planner={planner_id}, plan_only={self._plan_only})"
        )
        self.get_logger().info(
            f"Listening for target poses on {target_pose_topic}. "
            "Next step is wiring this node to MoveIt's action or planning interface."
        )
        self.get_logger().info(
            f"Listening for selected item points on {selected_item_point_topic}; "
            f"planning points on {selected_item_moveit_point_topic}; "
            f"first world point is latched and republished on {latched_target_point_topic}. "
            f"Publish std_msgs/Empty on {clear_latched_target_topic} to latch a new point."
        )
        if not MOVEIT_ACTIONS_AVAILABLE:
            self.get_logger().warning(
                "moveit_msgs is not importable in this environment. The planner will "
                "publish/latch grasp poses but cannot send MoveGroup execution goals here."
            )

    def _validate_external_dependencies(self) -> None:
        missing = []
        for dependency in REQUIRED_EXTERNAL_PACKAGES:
            try:
                get_package_share_directory(dependency.package_name)
            except PackageNotFoundError:
                missing.append(dependency)

        if not missing:
            self.get_logger().info("Detected MoveIt and cuMotion packages in the environment")
            return

        missing_text = ", ".join(
            f"{dependency.package_name} ({dependency.purpose})" for dependency in missing
        )
        self.get_logger().warning(
            "Missing external motion dependencies: "
            f"{missing_text}. Build the container image again after installing "
            "MoveIt and Isaac ROS cuMotion packages."
        )

    def _float_list_parameter(self, name: str, expected_length: int) -> list[float]:
        value = self.get_parameter(name).value
        if not isinstance(value, (list, tuple)) or len(value) != expected_length:
            raise ValueError(
                f"Parameter {name!r} must be a list of {expected_length} numbers"
            )
        return [float(element) for element in value]

    def _handle_target_pose(self, message: PoseStamped) -> None:
        self.get_logger().info(
            "Received target pose in frame "
            f"{message.header.frame_id!r} at "
            f"({message.pose.position.x:.3f}, "
            f"{message.pose.position.y:.3f}, "
            f"{message.pose.position.z:.3f})."
        )

    def _handle_selected_item_point(self, message: PointStamped) -> None:
        self._latest_world_selected_item_point = deepcopy(message)

    def _handle_joint_state(self, message: JointState) -> None:
        if not message.name or len(message.name) != len(message.position):
            return
        self._latest_joint_state = deepcopy(message)
        self._joint_state_ready = True
        if self._logged_joint_state_names:
            return
        self._logged_joint_state_names = True
        self.get_logger().info(
            "Received first valid /joint_states message with joints: "
            + ", ".join(message.name)
        )

    def _handle_selected_item_moveit_point(self, message: PointStamped) -> None:
        if self._latched_target_point is not None:
            return

        self._latched_target_point = deepcopy(
            self._latest_world_selected_item_point or message
        )
        self._latched_target_point_publisher.publish(self._latched_target_point)
        self.get_logger().info(
            "Latched selected item point for visualization in frame "
            f"{self._latched_target_point.header.frame_id!r} at "
            f"({self._latched_target_point.point.x:.3f}, "
            f"{self._latched_target_point.point.y:.3f}, "
            f"{self._latched_target_point.point.z:.3f}); "
            f"planning in frame {message.header.frame_id!r} at "
            f"({message.point.x:.3f}, {message.point.y:.3f}, {message.point.z:.3f})."
        )
        self._publish_grasp_debug_poses_in_moveit_frame(message)
        if self._auto_execute_pick:
            if not self._joint_state_ready or self._latest_joint_state is None:
                self.get_logger().warning(
                    "Deferring pick start until a valid /joint_states message is available."
                )
                return
            self._start_pick_sequence(message)

    def _handle_clear_latched_target(self, _message: Empty) -> None:
        if self._latched_target_point is None:
            self.get_logger().info("No latched target point to clear.")
            self._abort_pick_sequence("clear requested")
            return

        point = self._latched_target_point.point
        self._latched_target_point = None
        self._abort_pick_sequence("clear requested")
        self.get_logger().info(
            "Cleared latched target point at "
            f"({point.x:.3f}, {point.y:.3f}, {point.z:.3f})."
        )

    def _publish_grasp_debug_poses_in_moveit_frame(self, moveit_point: PointStamped) -> None:
        pregrasp_pose, grasp_pose, retract_pose, drop_pose = self._build_grasp_poses(
            moveit_point,
        )
        self._pregrasp_pose_publisher.publish(pregrasp_pose)
        self._grasp_pose_publisher.publish(grasp_pose)
        self._retract_pose_publisher.publish(retract_pose)
        self._drop_pose_publisher.publish(drop_pose)

    def _build_grasp_poses(
        self,
        point: PointStamped,
    ) -> tuple[PoseStamped, PoseStamped, PoseStamped, PoseStamped]:
        grasp_pose = self._pose_from_point_offset(point, self._grasp_offset_xyz)
        pregrasp_pose = self._pose_from_point_offset(point, self._pregrasp_offset_xyz)
        retract_pose = deepcopy(pregrasp_pose)
        drop_pose = self._pose_from_xyz(
            point.header.frame_id,
            point.header.stamp,
            self._drop_position_xyz,
        )
        self._set_top_down_orientation(drop_pose)
        return pregrasp_pose, grasp_pose, retract_pose, drop_pose

    def _pose_from_point_offset(
        self,
        point: PointStamped,
        offset_xyz: list[float],
    ) -> PoseStamped:
        return self._pose_from_xyz(
            point.header.frame_id,
            point.header.stamp,
            [
                point.point.x + offset_xyz[0],
                point.point.y + offset_xyz[1],
                point.point.z + offset_xyz[2],
            ],
        )

    def _offset_pose(self, pose: PoseStamped, offset_xyz: list[float]) -> PoseStamped:
        offset_pose = deepcopy(pose)
        offset_pose.pose.position.x += offset_xyz[0]
        offset_pose.pose.position.y += offset_xyz[1]
        offset_pose.pose.position.z += offset_xyz[2]
        return offset_pose

    def _pose_from_xyz(self, frame_id: str, stamp: Any, xyz: list[float]) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp
        pose.pose.position.x = xyz[0]
        pose.pose.position.y = xyz[1]
        pose.pose.position.z = xyz[2]
        pose.pose.orientation.w = 1.0
        return pose

    def _set_top_down_orientation(self, pose: PoseStamped) -> None:
        pose.pose.orientation.x = self._top_down_orientation_xyzw[0]
        pose.pose.orientation.y = self._top_down_orientation_xyzw[1]
        pose.pose.orientation.z = self._top_down_orientation_xyzw[2]
        pose.pose.orientation.w = self._top_down_orientation_xyzw[3]

    def _start_pick_sequence(self, point: PointStamped) -> None:
        if self._pick_in_progress:
            self.get_logger().warning("Pick sequence is already running; ignoring target.")
            return

        pregrasp_pose, grasp_pose, retract_pose, drop_pose = self._build_grasp_poses(point)
        self._pick_sequence = [
            MotionStep("approach pregrasp", pose=pregrasp_pose),
            MotionStep("move to grasp", pose=grasp_pose),
            MotionStep("close right gripper", gripper_closed=True),
            MotionStep("retract", pose=retract_pose),
            MotionStep(
                "move to drop",
                pose=drop_pose,
                force_top_down=True,
                top_down_tolerance_xyz=self._release_top_down_orientation_tolerance_xyz,
                top_down_weight=self._release_top_down_orientation_weight,
            ),
            MotionStep("open right gripper", gripper_closed=False),
        ]
        self._pick_sequence_index = 0
        self._pick_generation += 1
        self._pick_in_progress = True
        self.get_logger().info("Starting fixed-scene right-arm pick sequence.")
        self._execute_next_pick_step()

    def _abort_pick_sequence(self, reason: str) -> None:
        self._pick_generation += 1
        self._pick_in_progress = False
        self._pick_sequence = []
        self._pick_sequence_index = 0
        if self._step_timer is not None:
            self._step_timer.cancel()
            self._step_timer = None
        if self._move_result_timer is not None:
            self._move_result_timer.cancel()
            self._move_result_timer = None
        if self._active_goal_handle is not None:
            self.get_logger().info(
                f"Stopped local pick sequence after {reason}; not canceling active "
                "MoveGroup goal because MoveGroup can crash on preempt in this setup."
            )
            self._active_goal_handle = None

    def _execute_next_pick_step(self) -> None:
        if self._pick_sequence_index >= len(self._pick_sequence):
            self._pick_in_progress = False
            self.get_logger().info("Pick sequence completed.")
            return

        step = self._pick_sequence[self._pick_sequence_index]
        self._pick_sequence_index += 1

        if step.gripper_closed is not None:
            self._publish_right_gripper_command(step.gripper_closed)
            state = "closed" if step.gripper_closed else "opened"
            self.get_logger().info(
                f"Marked right gripper {state}. "
                "Published direct joint command to Isaac Sim."
            )
            self._step_timer = self.create_timer(
                self._gripper_command_settle_seconds,
                self._execute_next_pick_step_once,
            )
            return

        if step.pose is None:
            self.get_logger().error(f"Pick step {step.name!r} has no target pose.")
            self._pick_in_progress = False
            return

        if not MOVEIT_ACTIONS_AVAILABLE or self._move_group_client is None:
            self.get_logger().warning(
                f"Cannot execute {step.name!r}; MoveGroup action support is unavailable."
            )
            self._pick_in_progress = False
            return
        if self._move_group_goal_may_be_active:
            self.get_logger().error(
                "Not sending a new MoveGroup goal because the previous goal may still "
                "be active inside MoveGroup. Restart move_group before retrying."
            )
            self._pick_in_progress = False
            return

        if not self._move_group_client.server_is_ready():
            self.get_logger().info("Waiting for MoveGroup action server...")
            if not self._move_group_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("MoveGroup action server is not available.")
                self._pick_in_progress = False
                return

        self.get_logger().info(
            f"Sending MoveGroup {'plan' if self._plan_only else 'goal'} for {step.name!r} at "
            f"({step.pose.pose.position.x:.3f}, "
            f"{step.pose.pose.position.y:.3f}, "
            f"{step.pose.pose.position.z:.3f}) in {step.pose.header.frame_id!r}."
        )
        generation = self._pick_generation
        send_future = self._move_group_client.send_goal_async(
            self._build_move_group_goal(step)
        )
        send_future.add_done_callback(
            lambda future, current_step=step, current_generation=generation: self._handle_move_goal_response(
                future,
                current_step,
                current_generation,
            )
        )

    def _execute_next_pick_step_once(self) -> None:
        if self._step_timer is not None:
            self._step_timer.cancel()
            self._step_timer = None
        self._execute_next_pick_step()

    def _build_move_group_goal(self, step: MotionStep) -> Any:
        if step.pose is None:
            raise RuntimeError(f"Cannot build MoveGroup goal for {step.name!r} without pose.")

        goal = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = self._right_arm_group
        request.pipeline_id = self._pipeline_id
        request.planner_id = self._planner_id
        request.num_planning_attempts = self._num_planning_attempts
        request.allowed_planning_time = self._allowed_planning_time
        request.max_velocity_scaling_factor = self._max_velocity_scaling_factor
        request.max_acceleration_scaling_factor = self._max_acceleration_scaling_factor
        request.start_state = self._build_start_state()
        request.goal_constraints = [self._build_pose_constraints(step)]

        options = PlanningOptions()
        options.plan_only = self._plan_only
        options.look_around = False
        options.replan = True
        options.planning_scene_diff.is_diff = True
        options.planning_scene_diff.robot_state.is_diff = True

        goal.request = request
        goal.planning_options = options
        return goal

    def _publish_right_gripper_command(self, closed: bool) -> None:
        joint_command = JointState()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.name = [
            self._right_gripper_joint,
            self._right_gripper_mimic_joint,
        ]
        position = (
            self._right_gripper_closed_position
            if closed
            else self._right_gripper_open_position
        )
        joint_command.position = [position, position]
        self._joint_command_publisher.publish(joint_command)

    def _build_start_state(self) -> Any:
        if self._latest_joint_state is None:
            raise RuntimeError("Cannot build MoveGroup start_state without /joint_states.")
        start_state = RobotState()
        start_state.is_diff = False
        start_state.joint_state = deepcopy(self._latest_joint_state)
        return start_state

    def _build_pose_constraints(self, step: MotionStep) -> Any:
        if step.pose is None:
            raise RuntimeError(f"Cannot build pose constraints for {step.name!r} without pose.")

        pose = step.pose
        constraints = Constraints()
        constraints.name = "right_arm_grasp_pose"

        position_constraint = PositionConstraint()
        position_constraint.header = pose.header
        position_constraint.link_name = self._end_effector_link
        position_constraint.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self._position_tolerance]
        region_pose = deepcopy(pose.pose)
        position_constraint.constraint_region = BoundingVolume()
        position_constraint.constraint_region.primitives.append(sphere)
        position_constraint.constraint_region.primitive_poses.append(region_pose)

        constraints.position_constraints.append(position_constraint)
        if self._use_top_down_orientation_constraint or step.force_top_down:
            tolerance_xyz = (
                step.top_down_tolerance_xyz
                if step.force_top_down and step.top_down_tolerance_xyz is not None
                else self._top_down_orientation_tolerance_xyz
            )
            weight = (
                step.top_down_weight
                if step.force_top_down
                else 0.5
            )
            orientation_constraint = OrientationConstraint()
            orientation_constraint.header = pose.header
            orientation_constraint.link_name = self._end_effector_link
            orientation_constraint.orientation.x = self._top_down_orientation_xyzw[0]
            orientation_constraint.orientation.y = self._top_down_orientation_xyzw[1]
            orientation_constraint.orientation.z = self._top_down_orientation_xyzw[2]
            orientation_constraint.orientation.w = self._top_down_orientation_xyzw[3]
            orientation_constraint.absolute_x_axis_tolerance = tolerance_xyz[0]
            orientation_constraint.absolute_y_axis_tolerance = tolerance_xyz[1]
            orientation_constraint.absolute_z_axis_tolerance = tolerance_xyz[2]
            orientation_constraint.weight = weight
            constraints.orientation_constraints.append(orientation_constraint)
        return constraints

    def _handle_move_goal_response(
        self,
        future: Any,
        step: MotionStep,
        generation: int,
    ) -> None:
        if generation != self._pick_generation:
            return
        try:
            goal_handle = future.result()
        except Exception as error:
            if generation == self._pick_generation:
                self.get_logger().error(
                    f"MoveGroup goal response failed for pick step {step.name!r}: {error}"
                )
                self._pick_in_progress = False
            return
        if not goal_handle.accepted:
            self.get_logger().error(f"MoveGroup rejected pick step {step.name!r}.")
            self._pick_in_progress = False
            return

        self.get_logger().info(f"MoveGroup accepted pick step {step.name!r}; waiting for result.")
        self._active_goal_handle = goal_handle
        self._move_group_goal_may_be_active = True
        self._move_result_timer = self.create_timer(
            self._move_group_result_timeout,
            lambda current_step=step, current_generation=generation: self._handle_move_result_timeout(
                current_step,
                current_generation,
            ),
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, current_step=step, current_generation=generation: self._handle_move_result(
                result,
                current_step,
                current_generation,
            )
        )

    def _handle_move_result(self, future: Any, step: MotionStep, generation: int) -> None:
        self._move_group_goal_may_be_active = False
        if self._move_result_timer is not None:
            self._move_result_timer.cancel()
            self._move_result_timer = None
        if generation != self._pick_generation:
            return
        self._active_goal_handle = None
        try:
            action_result = future.result()
            result = action_result.result
        except Exception as error:
            self.get_logger().error(
                f"MoveGroup result failed for pick step {step.name!r}: {error}"
            )
            self._pick_in_progress = False
            return
        action_status = getattr(action_result, "status", None)
        error_code = getattr(getattr(result, "error_code", None), "val", 1)
        error_name = MOVEIT_ERROR_CODE_NAMES.get(error_code, f"UNKNOWN_{error_code}")
        planned_points = self._trajectory_point_count(
            getattr(result, "planned_trajectory", None)
        )
        executed_points = self._trajectory_point_count(
            getattr(result, "executed_trajectory", None)
        )
        planning_time = float(getattr(result, "planning_time", 0.0))
        planned_final = self._trajectory_final_joint_summary(
            getattr(result, "planned_trajectory", None)
        )
        executed_final = self._trajectory_final_joint_summary(
            getattr(result, "executed_trajectory", None)
        )
        if error_code != 1:
            self.get_logger().error(
                f"MoveGroup failed pick step {step.name!r}: "
                f"error={error_name} ({error_code}), action_status={action_status}, "
                f"planning_time={planning_time:.3f}s, "
                f"planned_points={planned_points}, executed_points={executed_points}."
            )
            self._pick_in_progress = False
            return

        self.get_logger().info(
            f"MoveGroup completed pick step {step.name!r}: "
            f"error={error_name} ({error_code}), action_status={action_status}, "
            f"planning_time={planning_time:.3f}s, "
            f"planned_points={planned_points}, executed_points={executed_points}."
        )
        if planned_final:
            self.get_logger().info(
                f"MoveGroup planned final joints for {step.name!r}: {planned_final}"
            )
        if executed_final:
            self.get_logger().info(
                f"MoveGroup executed final joints for {step.name!r}: {executed_final}"
            )
        self._execute_next_pick_step()

    def _handle_move_result_timeout(self, step: MotionStep, generation: int) -> None:
        if self._move_result_timer is not None:
            self._move_result_timer.cancel()
            self._move_result_timer = None
        if generation != self._pick_generation:
            return
        self._pick_generation += 1
        self._pick_in_progress = False
        self._active_goal_handle = None
        self.get_logger().error(
            f"MoveGroup did not return a result for pick step {step.name!r} within "
            f"{self._move_group_result_timeout:.1f}s. The goal may still be active "
            "inside MoveGroup; not canceling it because preempt can crash this setup. "
            "Restart move_group before retrying."
        )

    def _trajectory_point_count(self, trajectory: Any | None) -> int:
        if trajectory is None:
            return 0
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        points = getattr(joint_trajectory, "points", None)
        if points is None:
            return 0
        return len(points)

    def _trajectory_final_joint_summary(self, trajectory: Any | None) -> str:
        if trajectory is None:
            return ""
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        if joint_trajectory is None:
            return ""
        joint_names = list(getattr(joint_trajectory, "joint_names", []))
        points = list(getattr(joint_trajectory, "points", []))
        if not joint_names or not points:
            return ""
        positions = list(getattr(points[-1], "positions", []))
        if not positions:
            return ""
        return ", ".join(
            f"{joint_name}={position:.4f}"
            for joint_name, position in zip(joint_names, positions)
        )


def main() -> None:
    rclpy.init()
    node = MotionPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
