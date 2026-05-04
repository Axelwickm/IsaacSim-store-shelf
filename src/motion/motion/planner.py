#!/usr/bin/env python3

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PointStamped, Pose, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .arm_config import JOINT_LIMIT_MARGIN_RAD, JOINT_POSITION_LIMITS, get_arm_side_config
from .coordination import decode_payload, encode_payload
from .occupancy import (
    build_other_arm_collision_objects,
    is_arm_link,
    merged_position_map,
    sample_peer_plan_point,
    serialize_planned_trajectory,
)

try:
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        AllowedCollisionEntry,
        BoundingVolume,
        CollisionObject,
        Constraints,
        DisplayTrajectory,
        MotionPlanRequest,
        OrientationConstraint,
        PlanningSceneComponents,
        PlanningOptions,
        PlanningScene,
        PositionConstraint,
        RobotState,
    )
    from moveit_msgs.srv import GetPlanningScene
    from rclpy.action import ActionClient
    from shape_msgs.msg import SolidPrimitive

    MOVEIT_ACTIONS_AVAILABLE = True
except ModuleNotFoundError:
    AllowedCollisionEntry = None
    CollisionObject = None
    Duration = None
    FollowJointTrajectory = None
    DisplayTrajectory = None
    MoveGroup = None
    GetPlanningScene = None
    PlanningSceneComponents = None
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

RELEASE_TARGET_ON_MOVEIT_ERROR_CODES = {
    99999,  # Generic planning failure from MoveIt/cuMotion.
    -1,     # PLANNING_FAILED
    -2,     # INVALID_MOTION_PLAN
    -3,     # MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE
    -10,    # START_STATE_IN_COLLISION
    -11,    # START_STATE_VIOLATES_PATH_CONSTRAINTS
    -12,    # GOAL_IN_COLLISION
    -13,    # GOAL_VIOLATES_PATH_CONSTRAINTS
    -14,    # GOAL_CONSTRAINTS_VIOLATED
    -16,    # INVALID_GOAL_CONSTRAINTS
    -21,    # FRAME_TRANSFORM_FAILURE
    -22,    # COLLISION_CHECKING_UNAVAILABLE
    -29,    # NO_IK_SOLUTION
}

MOVE_GROUP_SERVER_WAIT_SECONDS = 5.0
MOVE_GROUP_SERVER_RETRY_SECONDS = 2.0


class MotionPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_planner")
        self.declare_parameter("planning_group", "yumi_arm")
        self.declare_parameter("move_group_name", "")
        self.declare_parameter("planning_arm_side", "right")
        self.declare_parameter("end_effector_link", "")
        self.declare_parameter("moveit_target_frame", "yumi_body")
        self.declare_parameter("pipeline_id", "isaac_ros_cumotion")
        self.declare_parameter("planner_id", "cuMotion")
        self.declare_parameter("move_group_action", "/move_action")
        self.declare_parameter(
            "direct_trajectory_action",
            "",
        )
        self.declare_parameter("direct_trajectory_result_timeout", 30.0)
        self.declare_parameter("direct_trajectory_goal_time_tolerance", 30.0)
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("gripper_joint", "")
        self.declare_parameter("gripper_mimic_joint", "")
        self.declare_parameter("gripper_open_position", 0.025)
        self.declare_parameter("gripper_closed_position", 0.0)
        self.declare_parameter("gripper_command_settle_seconds", 0.0)
        self.declare_parameter(
            "display_trajectory_topic",
            "/moveit/display_planned_path",
        )
        self.declare_parameter("plan_only", True)
        self.declare_parameter("pregrasp_offset_xyz", [0.0, 0.0, 0.08])
        self.declare_parameter("grasp_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("retract_offset_xyz", [0.0, 0.0, 0.20])
        self.declare_parameter("drop_position_xyz", [0.25, -0.14, 0.0])
        self.declare_parameter("left_drop_position_xyz", [0.25, 0.14, 0.0])
        self.declare_parameter("right_drop_position_xyz", [0.25, -0.14, 0.0])
        self.declare_parameter("position_tolerance", 0.025)
        self.declare_parameter("use_top_down_orientation_constraint", True)
        self.declare_parameter("top_down_orientation_xyzw", [0.0, 1.0, 0.0, 0.0])
        self.declare_parameter("top_down_orientation_tolerance_xyz", [0.9, 0.9, 3.14159])
        self.declare_parameter(
            "release_top_down_orientation_tolerance_xyz",
            [0.9, 0.9, 3.14159],
        )
        self.declare_parameter("release_top_down_orientation_weight", 0.5)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("move_group_result_timeout", 30.0)
        self.declare_parameter("move_group_observability_interval", 5.0)
        self.declare_parameter("move_group_debug_dump_dir", "/tmp/motion_planner_debug")
        self.declare_parameter("planning_scene_service", "/get_planning_scene")
        self.declare_parameter("planning_scene_topic", "/planning_scene")
        self.declare_parameter("robot_xrdf", "/workspace/usd/robot/yumi_isaacsim.xrdf")
        self.declare_parameter("robot_urdf", "/workspace/usd/robot/yumi_isaacsim.urdf")
        self.declare_parameter("planned_trajectory_collision_diagnostic", True)
        self.declare_parameter("inject_other_arm_collision_objects", True)
        self.declare_parameter("other_arm_collision_radius_padding", 0.03)
        self.declare_parameter("other_arm_sweep_sample_count", 8)
        self.declare_parameter(
            "required_planning_scene_object_ids",
            [
                "shelf_board_0",
                "shelf_board_1",
                "shelf_board_3",
                "shelf_board_4",
                "shelf_back_panel",
            ],
        )
        self.declare_parameter("planning_scene_ready_poll_period", 0.5)
        self.declare_parameter("num_planning_attempts", 5)
        self.declare_parameter("max_velocity_scaling_factor", 0.2)
        self.declare_parameter("max_acceleration_scaling_factor", 0.2)
        self.declare_parameter("arm_state_topic", "/motion/arm_state")
        self.declare_parameter("coordinator_state_topic", "/motion/coordinator_state")
        self.declare_parameter("reset_topic", "/motion/reset")
        self.declare_parameter("peer_plan_topic", "/motion/peer_plan")
        self.declare_parameter("peer_plan_extra_horizon_seconds", 2.0)

        self._validate_external_dependencies()

        planning_group = str(self.get_parameter("planning_group").value)
        planning_arm_side = str(self.get_parameter("planning_arm_side").value).strip().lower()
        self._planning_arm_side = planning_arm_side
        self._planning_arm_config = get_arm_side_config(self._planning_arm_side)
        self._other_arm_side = "left" if self._planning_arm_side == "right" else "right"
        self._other_arm_config = get_arm_side_config(self._other_arm_side)
        self._planning_joint_order = self._planning_arm_config.joint_order
        self._planning_start_joints = self._planning_arm_config.planning_start_joints
        configured_move_group_name = str(self.get_parameter("move_group_name").value).strip()
        self._move_group_name = (
            configured_move_group_name
            or self._planning_arm_config.move_group_name
        )
        configured_end_effector_link = str(
            self.get_parameter("end_effector_link").value
        ).strip()
        self._end_effector_link = (
            configured_end_effector_link
            or self._planning_arm_config.end_effector_link
        )
        self._moveit_target_frame = str(self.get_parameter("moveit_target_frame").value)
        pipeline_id = str(self.get_parameter("pipeline_id").value)
        planner_id = str(self.get_parameter("planner_id").value)
        move_group_action = str(self.get_parameter("move_group_action").value)
        planning_scene_service = str(
            self.get_parameter("planning_scene_service").value
        )
        planning_scene_topic = str(self.get_parameter("planning_scene_topic").value)
        configured_direct_trajectory_action = str(
            self.get_parameter("direct_trajectory_action").value
        ).strip()
        direct_trajectory_action = (
            configured_direct_trajectory_action
            or self._planning_arm_config.direct_trajectory_action
        )
        joint_command_topic = str(self.get_parameter("joint_command_topic").value)
        display_trajectory_topic = str(
            self.get_parameter("display_trajectory_topic").value
        )
        configured_gripper_joint = str(self.get_parameter("gripper_joint").value).strip()
        self._gripper_joint = (
            configured_gripper_joint or self._planning_arm_config.gripper_joint
        )
        configured_gripper_mimic_joint = str(
            self.get_parameter("gripper_mimic_joint").value
        ).strip()
        self._gripper_mimic_joint = (
            configured_gripper_mimic_joint
            or self._planning_arm_config.gripper_mimic_joint
        )
        self._gripper_open_position = float(
            self.get_parameter("gripper_open_position").value
        )
        self._gripper_closed_position = float(
            self.get_parameter("gripper_closed_position").value
        )
        self._gripper_command_settle_seconds = float(
            self.get_parameter("gripper_command_settle_seconds").value
        )
        self._arm_state_topic = str(self.get_parameter("arm_state_topic").value)
        self._coordinator_state_topic = str(
            self.get_parameter("coordinator_state_topic").value
        )
        self._reset_topic = str(self.get_parameter("reset_topic").value)
        self._peer_plan_topic = str(self.get_parameter("peer_plan_topic").value)
        self._peer_plan_extra_horizon_seconds = max(
            float(self.get_parameter("peer_plan_extra_horizon_seconds").value),
            0.0,
        )

        self._assigned_world_point: PointStamped | None = None
        self._latest_joint_state: JointState | None = None
        self._joint_state_ready = False
        self._logged_joint_state_names = False
        self._step_running = False
        self._pending_pick_point: PointStamped | None = None
        self._pick_generation = 0
        self._move_request_serial = 0
        self._step_timer: Any | None = None
        self._move_result_timer: Any | None = None
        self._move_observability_timer: Any | None = None
        self._active_goal_handle: Any | None = None
        self._active_move_step_name = ""
        self._active_move_request_id = ""
        self._active_move_request_summary = ""
        self._active_move_request_profile: dict[str, Any] = {}
        self._active_move_started_monotonic = 0.0
        self._move_group_goal_may_be_active = False
        self._pipeline_id = pipeline_id
        self._planner_id = planner_id
        self._plan_only = bool(self.get_parameter("plan_only").value)
        self._direct_trajectory_result_timeout = max(
            float(self.get_parameter("direct_trajectory_result_timeout").value),
            0.0,
        )
        self._direct_trajectory_goal_time_tolerance = max(
            float(self.get_parameter("direct_trajectory_goal_time_tolerance").value),
            0.0,
        )
        self._pregrasp_offset_xyz = self._float_list_parameter("pregrasp_offset_xyz", 3)
        self._grasp_offset_xyz = self._float_list_parameter("grasp_offset_xyz", 3)
        self._retract_offset_xyz = self._float_list_parameter("retract_offset_xyz", 3)
        self._drop_position_xyz = self._drop_position_for_arm()
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
        self._move_group_result_timeout = max(
            float(self.get_parameter("move_group_result_timeout").value),
            self._allowed_planning_time,
        )
        self._move_group_observability_interval = max(
            float(self.get_parameter("move_group_observability_interval").value),
            0.0,
        )
        self._move_group_debug_dump_dir = Path(
            str(self.get_parameter("move_group_debug_dump_dir").value)
        )
        self._move_group_debug_dump_dir.mkdir(parents=True, exist_ok=True)
        self._robot_xrdf = Path(str(self.get_parameter("robot_xrdf").value))
        self._robot_urdf = Path(str(self.get_parameter("robot_urdf").value))
        self._planned_trajectory_collision_diagnostic = bool(
            self.get_parameter("planned_trajectory_collision_diagnostic").value
        )
        self._inject_other_arm_collision_objects = bool(
            self.get_parameter("inject_other_arm_collision_objects").value
        )
        self._other_arm_collision_radius_padding = max(
            float(self.get_parameter("other_arm_collision_radius_padding").value),
            0.0,
        )
        self._other_arm_sweep_sample_count = max(
            int(self.get_parameter("other_arm_sweep_sample_count").value),
            1,
        )
        self._required_planning_scene_object_ids = {
            str(object_id)
            for object_id in self.get_parameter(
                "required_planning_scene_object_ids"
            ).value
        }
        self._planning_scene_ready_poll_period = max(
            float(self.get_parameter("planning_scene_ready_poll_period").value),
            0.1,
        )
        self._num_planning_attempts = int(
            self.get_parameter("num_planning_attempts").value
        )
        self._max_velocity_scaling_factor = float(
            self.get_parameter("max_velocity_scaling_factor").value
        )
        self._max_acceleration_scaling_factor = float(
            self.get_parameter("max_acceleration_scaling_factor").value
        )
        self._collision_model = None
        self._collision_tensor_args = None
        self._collision_joint_names: list[str] = []
        self._collision_sphere_links: list[str] = []
        self._collision_sphere_labels: list[str] = []
        self._collision_ignored_link_pairs: set[frozenset[str]] = set()
        self._other_arm_proxy_links: set[str] = set()
        self._last_planned_collision_waypoint_index: int | None = None
        self._last_planned_collision_waypoint_joint_names: list[str] = []
        self._last_planned_collision_waypoint_positions: list[float] = []
        self._planning_scene_client: Any | None = None
        self._planning_scene_ready = not self._required_planning_scene_object_ids
        self._planning_scene_request_in_flight = False
        self._planning_scene_ready_logged = False
        self._dynamic_collision_object_ids: set[str] = set()
        self._pending_pick_retry_timer: Any | None = None
        self._coordinator_state: dict[str, Any] = {"arms": {}}
        self._current_plan_state: dict[str, Any] | None = None
        self._peer_plan_state: dict[str, Any] | None = None
        self._active_step_command_id = 0
        self._active_step_name = ""
        self._step_status: dict[str, Any] | None = None
        self._last_plan_outcome: dict[str, Any] | None = None
        self._joint_command_publisher = self.create_publisher(
            JointState,
            joint_command_topic,
            10,
        )
        self._arm_state_publisher = self.create_publisher(
            String,
            self._arm_state_topic,
            10,
        )
        self._peer_plan_publisher = self.create_publisher(
            String,
            self._peer_plan_topic,
            10,
        )
        self._planning_scene_diff_publisher = None
        if MOVEIT_ACTIONS_AVAILABLE:
            self._planning_scene_diff_publisher = self.create_publisher(
                PlanningScene,
                planning_scene_topic,
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
        self._display_trajectory_publisher = None
        if MOVEIT_ACTIONS_AVAILABLE:
            self._display_trajectory_publisher = self.create_publisher(
                DisplayTrajectory,
                display_trajectory_topic,
                10,
            )
        self._move_group_client: Any | None = None
        self._direct_trajectory_client: Any | None = None
        if MOVEIT_ACTIONS_AVAILABLE:
            self._move_group_client = ActionClient(self, MoveGroup, move_group_action)
            self._direct_trajectory_client = ActionClient(
                self,
                FollowJointTrajectory,
                direct_trajectory_action,
            )
            self._planning_scene_client = self.create_client(
                GetPlanningScene,
                planning_scene_service,
            )

        self._joint_state_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self._handle_joint_state,
            10,
        )
        self._coordinator_state_subscription = self.create_subscription(
            String,
            self._coordinator_state_topic,
            self._handle_coordinator_state,
            10,
        )
        self._reset_subscription = self.create_subscription(
            String,
            self._reset_topic,
            self._handle_reset,
            10,
        )
        self._peer_plan_subscription = self.create_subscription(
            String,
            self._peer_plan_topic,
            self._handle_peer_plan,
            10,
        )

        self.get_logger().info(
            "Motion planner scaffold ready "
            f"(group={planning_group}, move_group={self._move_group_name}, "
            f"planning_arm_side={self._planning_arm_side}, "
            f"target_frame={self._moveit_target_frame}, "
            f"pipeline={pipeline_id}, planner={planner_id}, "
            f"direct_execute={not self._plan_only})"
        )
        self.get_logger().info(
            "Waiting for assigned targets from the motion coordinator."
        )
        self.get_logger().info(
            f"MoveGroup debug artifacts will be written under "
            f"{self._move_group_debug_dump_dir}"
        )
        self.get_logger().info(
            f"Arm state topic: {self._arm_state_topic}; "
            f"coordinator state topic: {self._coordinator_state_topic}; "
            f"reset topic: {self._reset_topic}; "
            f"peer plan topic: {self._peer_plan_topic}"
        )
        if self._required_planning_scene_object_ids:
            self.get_logger().info(
                "Will defer planning until MoveIt planning scene contains: "
                + ", ".join(sorted(self._required_planning_scene_object_ids))
            )
        if not MOVEIT_ACTIONS_AVAILABLE:
            self.get_logger().warning(
                "moveit_msgs is not importable in this environment. The planner will "
                "publish/latch grasp poses but cannot send MoveGroup execution goals here."
            )
        self._publish_arm_state()

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

    def _latch_assigned_target(
        self,
        world_point: PointStamped,
        moveit_point: PointStamped,
    ) -> None:
        if self._assigned_world_point is not None:
            return
        self._assigned_world_point = deepcopy(world_point)
        self._pending_pick_point = deepcopy(moveit_point)
        self._publish_arm_state()
        self.get_logger().info(
            "Accepted assigned target from coordinator in frame "
            f"{self._assigned_world_point.header.frame_id!r} at "
            f"({self._assigned_world_point.point.x:.3f}, "
            f"{self._assigned_world_point.point.y:.3f}, "
            f"{self._assigned_world_point.point.z:.3f}); "
            f"planning in frame {moveit_point.header.frame_id!r} at "
            f"({moveit_point.point.x:.3f}, {moveit_point.point.y:.3f}, "
            f"{moveit_point.point.z:.3f})."
        )
        self._publish_grasp_debug_poses_in_moveit_frame(moveit_point)
        self._try_execute_assigned_step()

    def _release_latched_target(self, reason: str) -> None:
        if self._assigned_world_point is None:
            return

        point = self._assigned_world_point.point
        self._assigned_world_point = None
        self._pending_pick_point = None
        self._last_plan_outcome = None
        self._publish_arm_state()
        self.get_logger().info(
            f"Released {reason} target at "
            f"({point.x:.3f}, {point.y:.3f}, {point.z:.3f}); "
            "waiting for the next coordinator assignment."
        )

    def _handle_coordinator_state(self, message: String) -> None:
        payload = decode_payload(message)
        if payload is None:
            self.get_logger().warning(
                "Ignoring invalid coordinator state payload.",
                throttle_duration_sec=5.0,
            )
            return
        self._coordinator_state = payload
        self._sync_assigned_target_from_coordinator()
        self._try_execute_assigned_step()

    def _handle_peer_plan(self, message: String) -> None:
        payload = decode_payload(message)
        if payload is None:
            return
        arm_side = str(payload.get("arm_side", "")).strip().lower()
        if arm_side != self._other_arm_side:
            return
        self._peer_plan_state = payload

    def _handle_reset(self, message: String) -> None:
        payload = decode_payload(message)
        reset_reason = ""
        if isinstance(payload, dict):
            reset_reason = str(payload.get("reason", "")).strip()
        self._clear_for_episode_reset()
        self.get_logger().info(
            "Received motion reset"
            + (f" reason={reset_reason}" if reset_reason else "")
        )

    def _sync_assigned_target_from_coordinator(self) -> None:
        assigned_target = self._assigned_target_for_self()
        if assigned_target is None:
            self._clear_local_assignment_if_idle()
            return
        if self._assigned_world_point is not None:
            return
        if self._step_running or self._move_group_goal_may_be_active:
            return
        self._clear_local_assignment_if_idle()
        assigned_target = self._assigned_target_for_self()
        if assigned_target is None:
            return
        world_point = self._deserialize_point_payload(assigned_target.get("world_point"))
        moveit_point = self._deserialize_point_payload(assigned_target.get("moveit_point"))
        if world_point is None or moveit_point is None:
            return
        self._latch_assigned_target(world_point, moveit_point)

    def _clear_local_assignment_if_idle(self) -> None:
        if self._assigned_world_point is None:
            return
        if self._step_running or self._move_group_goal_may_be_active:
            return
        self._assigned_world_point = None
        self._pending_pick_point = None
        self._publish_arm_state()

    def _clear_for_episode_reset(self) -> None:
        self._pick_generation += 1
        if self._pending_pick_retry_timer is not None:
            self._pending_pick_retry_timer.cancel()
            self._pending_pick_retry_timer = None
        if self._step_timer is not None:
            self._step_timer.cancel()
            self._step_timer = None
        if self._move_result_timer is not None:
            self._move_result_timer.cancel()
            self._move_result_timer = None
        if self._move_observability_timer is not None:
            self._move_observability_timer.cancel()
            self._move_observability_timer = None
        self._assigned_world_point = None
        self._pending_pick_point = None
        self._coordinator_state = {"arms": {}}
        self._current_plan_state = None
        self._peer_plan_state = None
        self._step_status = None
        self._last_plan_outcome = None
        self._step_running = False
        self._move_group_goal_may_be_active = False
        self._active_goal_handle = None
        self._active_step_command_id = 0
        self._active_step_name = ""
        self._active_move_started_monotonic = 0.0
        self._clear_active_move_observability()
        self._publish_arm_state()

    def _publish_arm_state(self) -> None:
        self._arm_state_publisher.publish(encode_payload(self._arm_state_payload()))

    def _arm_state_payload(self) -> dict[str, Any]:
        return {
            "arm_side": self._planning_arm_side,
            "stamp_ns": int(self.get_clock().now().nanoseconds),
            "assigned_target": self._assigned_target_for_self(),
            "plan_outcome": self._last_plan_outcome,
            "step_status": self._step_status,
            "motion_active": bool(self._move_group_goal_may_be_active),
        }

    def _peer_state(self) -> dict[str, Any]:
        arms = self._coordinator_state.get("arms")
        if not isinstance(arms, dict):
            return {}
        peer_state = arms.get(self._other_arm_side)
        return peer_state if isinstance(peer_state, dict) else {}

    def _publish_peer_plan_state(self) -> None:
        if self._current_plan_state is None:
            return
        payload = dict(self._current_plan_state)
        payload["arm_side"] = self._planning_arm_side
        self._peer_plan_publisher.publish(encode_payload(payload))

    def _assigned_target_for_self(self) -> dict[str, Any] | None:
        assigned_target = self._coordinator_state.get("assigned_target")
        if not isinstance(assigned_target, dict):
            return None
        assigned_arm = str(assigned_target.get("arm_side", "")).strip().lower()
        if assigned_arm != self._planning_arm_side:
            return None
        return assigned_target

    def _assigned_step_for_self(self) -> dict[str, Any] | None:
        active_step = self._coordinator_state.get("active_step")
        if not isinstance(active_step, dict):
            return None
        assigned_arm = str(active_step.get("arm_side", "")).strip().lower()
        if assigned_arm != self._planning_arm_side:
            return None
        return active_step

    def _deserialize_point_payload(self, payload: Any) -> PointStamped | None:
        if not isinstance(payload, dict):
            return None
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

    def _publish_grasp_debug_poses_in_moveit_frame(self, moveit_point: PointStamped) -> None:
        pregrasp_pose, grasp_pose, retract_pose, drop_pose = self._build_grasp_poses(
            moveit_point,
        )
        self._pregrasp_pose_publisher.publish(pregrasp_pose)
        self._grasp_pose_publisher.publish(grasp_pose)
        self._retract_pose_publisher.publish(retract_pose)
        self._drop_pose_publisher.publish(drop_pose)

    def _set_step_status(self, command_id: int, name: str, state: str) -> None:
        self._step_status = {
            "command_id": int(command_id),
            "name": str(name),
            "state": str(state),
        }
        self._publish_arm_state()

    def _record_plan_outcome(
        self,
        *,
        step_name: str,
        success: bool,
        request_id: str,
        error_code: int,
        error_name: str,
        action_status: Any,
        planning_time: float,
        planned_points: int,
    ) -> None:
        assigned_target = self._assigned_target_for_self() or {}
        self._last_plan_outcome = {
            "arm_side": self._planning_arm_side,
            "candidate_id": str(assigned_target.get("candidate_id", "")).strip(),
            "query_index": int(assigned_target.get("query_index") or -1),
            "step_name": str(step_name),
            "success": bool(success),
            "request_id": str(request_id),
            "error_code": int(error_code),
            "error_name": str(error_name),
            "action_status": action_status,
            "planning_time": float(planning_time),
            "planned_points": int(planned_points),
            "stamp_ns": int(self.get_clock().now().nanoseconds),
        }
        self._publish_arm_state()

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

    def _command_step_for_target(
        self,
        command_name: str,
        point: PointStamped,
    ) -> MotionStep | None:
        pregrasp_pose, grasp_pose, retract_pose, drop_pose = self._build_grasp_poses(point)
        if command_name == "approach_pregrasp":
            return MotionStep("approach pregrasp", pose=pregrasp_pose)
        if command_name == "move_to_grasp":
            return MotionStep("move to grasp", pose=grasp_pose)
        if command_name == "close_gripper":
            return MotionStep(f"close {self._planning_arm_side} gripper", gripper_closed=True)
        if command_name == "retract":
            return MotionStep("retract", pose=retract_pose)
        if command_name == "move_to_drop":
            return MotionStep(
                "move to drop",
                pose=drop_pose,
                force_top_down=True,
                top_down_tolerance_xyz=self._release_top_down_orientation_tolerance_xyz,
                top_down_weight=self._release_top_down_orientation_weight,
            )
        if command_name == "open_gripper":
            return MotionStep(f"open {self._planning_arm_side} gripper", gripper_closed=False)
        return None

    def _try_execute_assigned_step(self) -> None:
        if self._pending_pick_point is None or self._move_group_goal_may_be_active:
            return
        assigned_step = self._assigned_step_for_self()
        if assigned_step is None:
            return
        command_id = int(assigned_step.get("command_id") or 0)
        command_name = str(assigned_step.get("name", "")).strip()
        if command_id <= 0 or not command_name:
            return
        if (
            self._step_status is not None
            and int(self._step_status.get("command_id") or 0) == command_id
            and str(self._step_status.get("state", "")).strip().lower() in {"running", "succeeded", "failed"}
        ):
            return
        if not self._joint_state_ready or self._latest_joint_state is None:
            self.get_logger().warning(
                "Deferring step start until a valid /joint_states message is available."
            )
            self._schedule_pending_pick_retry()
            return
        if not self._planning_scene_ready:
            self._request_planning_scene_readiness_check()
            self.get_logger().warning(
                "Deferring step start until required planning scene objects are available."
            )
            self._schedule_pending_pick_retry()
            return
        self._cancel_pending_pick_retry()
        step = self._command_step_for_target(command_name, self._pending_pick_point)
        if step is None:
            self.get_logger().error(f"Unknown assigned step {command_name!r}.")
            self._set_step_status(command_id, command_name, "failed")
            return
        self._active_step_command_id = command_id
        self._active_step_name = command_name
        self._step_running = True
        self._set_step_status(command_id, command_name, "running")
        self._last_plan_outcome = None
        self._publish_arm_state()
        self.get_logger().info(
            f"Executing assigned step {command_name!r} for {self._planning_arm_side} arm "
            f"(command_id={command_id})."
        )
        self._execute_motion_step(step)

    def _drop_position_for_arm(self) -> list[float]:
        if self._planning_arm_side == "left":
            return self._float_list_parameter("left_drop_position_xyz", 3)
        if self._planning_arm_side == "right":
            return self._float_list_parameter("right_drop_position_xyz", 3)
        return self._float_list_parameter("drop_position_xyz", 3)

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

    def _schedule_pending_pick_retry(self) -> None:
        if self._pending_pick_retry_timer is not None:
            return
        self._pending_pick_retry_timer = self.create_timer(
            self._planning_scene_ready_poll_period,
            self._retry_pending_pick_once,
        )

    def _cancel_pending_pick_retry(self) -> None:
        if self._pending_pick_retry_timer is not None:
            self._pending_pick_retry_timer.cancel()
            self._pending_pick_retry_timer = None

    def _retry_pending_pick_once(self) -> None:
        self._cancel_pending_pick_retry()
        self._try_execute_assigned_step()

    def _peer_motion_active(self) -> bool:
        return bool(self._peer_state().get("motion_active"))

    def _request_planning_scene_readiness_check(self) -> None:
        if (
            self._planning_scene_ready
            or self._planning_scene_request_in_flight
            or self._planning_scene_client is None
        ):
            return
        if not self._planning_scene_client.service_is_ready():
            return
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        self._planning_scene_request_in_flight = True
        future = self._planning_scene_client.call_async(request)
        future.add_done_callback(self._handle_planning_scene_readiness_result)

    def _handle_planning_scene_readiness_result(self, future: Any) -> None:
        self._planning_scene_request_in_flight = False
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(
                f"Planning scene readiness check failed: {error}",
                throttle_duration_sec=5.0,
            )
            return
        object_ids = {
            collision_object.id
            for collision_object in response.scene.world.collision_objects
        }
        missing = sorted(self._required_planning_scene_object_ids - object_ids)
        if missing:
            self.get_logger().info(
                "Waiting for planning scene objects: " + ", ".join(missing),
                throttle_duration_sec=5.0,
            )
            return
        self._planning_scene_ready = True
        if not self._planning_scene_ready_logged:
            self._planning_scene_ready_logged = True
            self.get_logger().info(
                "Planning scene is ready; required shelf collision objects are present."
            )
        self._try_execute_assigned_step()

    def _execute_motion_step(self, step: MotionStep) -> None:
        if step.gripper_closed is not None:
            self._publish_gripper_command(step.gripper_closed)
            state = "closed" if step.gripper_closed else "opened"
            self.get_logger().info(
                f"Marked {self._planning_arm_side} gripper {state}. "
                "Published direct joint command to Isaac Sim."
            )
            if self._gripper_command_settle_seconds <= 0.0:
                self._step_running = False
                self._set_step_status(self._active_step_command_id, self._active_step_name, "succeeded")
            else:
                self._step_timer = self.create_timer(
                    self._gripper_command_settle_seconds,
                    self._complete_gripper_step_once,
                )
            return

        if step.pose is None:
            self.get_logger().error(f"Pick step {step.name!r} has no target pose.")
            self._fail_active_step()
            return

        if not MOVEIT_ACTIONS_AVAILABLE or self._move_group_client is None:
            self.get_logger().warning(
                f"Cannot execute {step.name!r}; MoveGroup action support is unavailable."
            )
            self._fail_active_step()
            return
        if self._move_group_goal_may_be_active:
            self.get_logger().error(
                "Not sending a new plan because a previous planning or execution goal "
                "may still be active. Restart the stack before retrying."
            )
            self._fail_active_step()
            return

        if not self._move_group_client.server_is_ready():
            self.get_logger().info("Waiting for MoveGroup action server...")
            if not self._move_group_client.wait_for_server(
                timeout_sec=MOVE_GROUP_SERVER_WAIT_SECONDS
            ):
                self.get_logger().warning(
                    "MoveGroup action server is not available yet; retrying pick step "
                    f"{step.name!r}."
                )
                self._retry_current_step()
                return
        if self._peer_motion_active():
            self.get_logger().info(
                "Peer arm has an active planning/execution goal; retrying current pick step."
            )
            self._retry_current_step()
            return

        try:
            goal = self._build_move_group_goal(step)
        except RuntimeError as error:
            self.get_logger().error(
                f"Cannot build MoveGroup goal for pick step {step.name!r}: {error}"
            )
            self._fail_active_step()
            return
        request_id, request_profile = self._build_move_group_observability(step, goal)
        request_summary = request_profile["summary"]
        self._write_move_group_debug_artifact(
            request_id,
            "sent",
            {
                "step_name": step.name,
                "profile": request_profile,
            },
        )
        self.get_logger().info(
            f"Sending MoveGroup plan request_id={request_id} for {step.name!r}: "
            f"{request_summary}"
        )
        generation = self._pick_generation
        send_future = self._move_group_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda future, current_step=step, current_generation=generation, current_request_id=request_id, current_request_summary=request_summary, current_request_profile=request_profile: self._handle_move_goal_response(
                future,
                current_step,
                current_generation,
                current_request_id,
                current_request_summary,
                current_request_profile,
            )
        )

    def _complete_gripper_step_once(self) -> None:
        if self._step_timer is not None:
            self._step_timer.cancel()
            self._step_timer = None
        self._step_running = False
        self._set_step_status(self._active_step_command_id, self._active_step_name, "succeeded")

    def _retry_current_step(self) -> None:
        self._step_timer = self.create_timer(
            MOVE_GROUP_SERVER_RETRY_SECONDS,
            self._retry_active_step_once,
        )

    def _fail_active_step(self) -> None:
        self._step_running = False
        self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")

    def _retry_active_step_once(self) -> None:
        if self._step_timer is not None:
            self._step_timer.cancel()
            self._step_timer = None
        self._step_running = False
        if self._step_status is not None:
            self._step_status = None
        self._publish_arm_state()
        self._try_execute_assigned_step()

    def _build_move_group_goal(self, step: MotionStep) -> Any:
        if step.pose is None:
            raise RuntimeError(f"Cannot build MoveGroup goal for {step.name!r} without pose.")

        goal = MoveGroup.Goal()
        goal.request = self._build_motion_plan_request(step)
        goal.planning_options = self._build_planning_options()
        return goal

    def _build_motion_plan_request(
        self,
        step: MotionStep,
        include_start_state: bool = True,
    ) -> Any:
        if step.pose is None:
            raise RuntimeError(f"Cannot build MotionPlanRequest for {step.name!r} without pose.")

        request = MotionPlanRequest()
        request.group_name = self._move_group_name
        request.pipeline_id = self._pipeline_id
        request.planner_id = self._planner_id
        request.num_planning_attempts = self._num_planning_attempts
        request.allowed_planning_time = self._allowed_planning_time
        request.max_velocity_scaling_factor = self._max_velocity_scaling_factor
        request.max_acceleration_scaling_factor = self._max_acceleration_scaling_factor
        if include_start_state:
            request.start_state = self._build_start_state()
        request.goal_constraints = [self._build_pose_constraints(step)]
        return request

    def _build_planning_options(self) -> Any:
        options = PlanningOptions()
        options.plan_only = True
        options.look_around = False
        options.replan = False
        options.planning_scene_diff.is_diff = True
        options.planning_scene_diff.robot_state.is_diff = True
        collision_objects = self._build_other_arm_collision_objects()
        if collision_objects:
            self._publish_dynamic_collision_scene(collision_objects)
            self._log_other_arm_proxy_start_state_collisions(collision_objects)
            options.planning_scene_diff.world.collision_objects.extend(collision_objects)
            self._allow_other_arm_proxy_self_overlap(
                options,
                collision_objects,
            )
        else:
            self._clear_dynamic_collision_scene()
        return options

    def _publish_dynamic_collision_scene(self, collision_objects: list[Any]) -> None:
        if self._planning_scene_diff_publisher is None or PlanningScene is None:
            return
        scene = PlanningScene()
        scene.is_diff = True
        remove_objects = []
        current_ids = {str(collision_object.id) for collision_object in collision_objects}
        for object_id in sorted(self._dynamic_collision_object_ids - current_ids):
            remove_object = CollisionObject()
            remove_object.header.frame_id = self._moveit_target_frame
            remove_object.id = object_id
            remove_object.operation = CollisionObject.REMOVE
            remove_objects.append(remove_object)
        scene.world.collision_objects = remove_objects + list(collision_objects)
        self._dynamic_collision_object_ids = current_ids
        self._planning_scene_diff_publisher.publish(scene)
        self.get_logger().info(
            f"Published {len(collision_objects)} dynamic other-arm collision objects "
            f"to MoveIt planning scene.",
            throttle_duration_sec=5.0,
        )

    def _clear_dynamic_collision_scene(self) -> None:
        if (
            self._planning_scene_diff_publisher is None
            or PlanningScene is None
            or CollisionObject is None
            or not self._dynamic_collision_object_ids
        ):
            return
        scene = PlanningScene()
        scene.is_diff = True
        for object_id in sorted(self._dynamic_collision_object_ids):
            collision_object = CollisionObject()
            collision_object.header.frame_id = self._moveit_target_frame
            collision_object.id = object_id
            collision_object.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(collision_object)
        self._dynamic_collision_object_ids = set()
        self._planning_scene_diff_publisher.publish(scene)

    def _is_planning_arm_link(self, link_name: str) -> bool:
        return is_arm_link(link_name, self._planning_arm_config)

    def _is_other_arm_link(self, link_name: str) -> bool:
        return is_arm_link(link_name, self._other_arm_config)

    def _log_other_arm_proxy_start_state_collisions(
        self,
        collision_objects: list[Any],
    ) -> None:
        if self._latest_joint_state is None or not collision_objects:
            return
        try:
            if self._collision_model is None or self._collision_tensor_args is None:
                self._load_collision_model()
        except Exception as error:
            self.get_logger().warn(
                f"Other-arm proxy start-state collision preflight unavailable: {error}",
                throttle_duration_sec=5.0,
            )
            return

        import torch

        position_map = self._collision_position_map()
        missing = [
            joint_name
            for joint_name in self._collision_joint_names
            if joint_name not in position_map
        ]
        if missing:
            self.get_logger().warn(
                "Skipping other-arm proxy start-state collision preflight; missing joint state for "
                + ", ".join(missing),
                throttle_duration_sec=5.0,
            )
            return

        q = torch.tensor(
            [[position_map[joint_name] for joint_name in self._collision_joint_names]],
            device=self._collision_tensor_args.device,
            dtype=self._collision_tensor_args.dtype,
        )
        spheres = (
            self._collision_model.get_state(q)
            .link_spheres_tensor[0]
            .detach()
            .cpu()
            .numpy()
        )

        planning_arm_spheres = []
        for index, sphere_values in enumerate(spheres):
            link_name = self._collision_sphere_links[index]
            if link_name != "yumi_body" and not self._is_planning_arm_link(link_name):
                continue
            x, y, z, radius = [float(value) for value in sphere_values]
            if radius <= 0.0:
                continue
            planning_arm_spheres.append(
                (
                    self._collision_sphere_labels[index],
                    x,
                    y,
                    z,
                    radius,
                )
            )

        proxy_spheres = []
        for collision_object in collision_objects:
            primitive = collision_object.primitives[0]
            pose = collision_object.primitive_poses[0]
            radius = float(primitive.dimensions[0])
            proxy_spheres.append(
                (
                    str(collision_object.id),
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    radius,
                )
            )

        overlaps: list[tuple[float, str, str, float, float]] = []
        for planning_label, planning_x, planning_y, planning_z, planning_radius in planning_arm_spheres:
            for proxy_id, proxy_x, proxy_y, proxy_z, proxy_radius in proxy_spheres:
                dx = planning_x - proxy_x
                dy = planning_y - proxy_y
                dz = planning_z - proxy_z
                distance = (dx * dx + dy * dy + dz * dz) ** 0.5
                radius_sum = planning_radius + proxy_radius
                penetration = radius_sum - distance
                if penetration > 0.0:
                    overlaps.append(
                        (
                            penetration,
                            planning_label,
                            proxy_id,
                            distance,
                            radius_sum,
                        )
                    )
        overlaps.sort(reverse=True)
        if not overlaps:
            self.get_logger().info(
                "Other-arm proxy start-state collision preflight: no planning-arm/proxy overlaps."
            )
            return
        penetration, planning_label, proxy_id, distance, radius_sum = overlaps[0]
        self.get_logger().warn(
            "Other-arm proxy start-state collision preflight: "
            f"{len(overlaps)} planning-arm/proxy overlaps. "
            f"Worst: {planning_label} <-> {proxy_id} "
            f"penetration={penetration:.4f}m distance={distance:.4f}m "
            f"radius_sum={radius_sum:.4f}m."
        )

    def _allow_other_arm_proxy_self_overlap(
        self,
        options: Any,
        collision_objects: list[Any],
    ) -> None:
        if AllowedCollisionEntry is None or not collision_objects:
            return
        acm = options.planning_scene_diff.allowed_collision_matrix
        acm.entry_names = list(acm.entry_names)
        acm.entry_values = list(acm.entry_values)

        proxy_ids = [str(collision_object.id) for collision_object in collision_objects]
        link_names = sorted(self._other_arm_proxy_links)
        if not proxy_ids or not link_names:
            return

        existing_names = list(acm.entry_names)
        name_to_index = {name: index for index, name in enumerate(existing_names)}

        def ensure_name(name: str) -> int:
            index = name_to_index.get(name)
            if index is not None:
                return index
            new_size = len(existing_names) + 1
            for entry in acm.entry_values:
                enabled = list(entry.enabled)
                enabled.append(False)
                entry.enabled = enabled
            new_entry = AllowedCollisionEntry()
            new_entry.enabled = [False] * new_size
            acm.entry_values.append(new_entry)
            existing_names.append(name)
            acm.entry_names.append(name)
            index = new_size - 1
            name_to_index[name] = index
            return index

        for link_name in link_names:
            ensure_name(link_name)
        for proxy_id in proxy_ids:
            ensure_name(proxy_id)

        for link_name in link_names:
            link_index = name_to_index[link_name]
            for proxy_id in proxy_ids:
                proxy_index = name_to_index[proxy_id]
                acm.entry_values[link_index].enabled[proxy_index] = True
                acm.entry_values[proxy_index].enabled[link_index] = True

    def _build_other_arm_collision_objects(self) -> list[Any]:
        if not self._inject_other_arm_collision_objects:
            return []
        if self._latest_joint_state is None:
            raise RuntimeError("latest joint state is unavailable")
        if CollisionObject is None:
            raise RuntimeError("MoveIt CollisionObject message support is unavailable")
        try:
            if self._collision_model is None or self._collision_tensor_args is None:
                self._load_collision_model()
        except Exception as error:
            raise RuntimeError(
                f"other-arm collision object injection unavailable: {error}"
            ) from error

        import torch

        self._log_peer_plan_debug_summary()
        collision_objects: list[Any] = []
        proxy_links: set[str] = set()
        sample_maps, proxy_source = self._peer_sweep_position_maps()
        for sample_index, position_map in enumerate(sample_maps):
            missing = [
                joint_name
                for joint_name in self._collision_joint_names
                if joint_name not in position_map
            ]
            if missing:
                self.get_logger().warn(
                    "Skipping other-arm collision object injection; missing joint state for "
                    + ", ".join(missing),
                    throttle_duration_sec=5.0,
                )
                return []

            q = torch.tensor(
                [[position_map[joint_name] for joint_name in self._collision_joint_names]],
                device=self._collision_tensor_args.device,
                dtype=self._collision_tensor_args.dtype,
            )
            spheres = (
                self._collision_model.get_state(q)
                .link_spheres_tensor[0]
                .detach()
                .cpu()
                .numpy()
            )
            sample_objects, sample_proxy_links = build_other_arm_collision_objects(
                spheres=spheres,
                sphere_links=self._collision_sphere_links,
                other_arm_config=self._other_arm_config,
                moveit_target_frame=self._moveit_target_frame,
                radius_padding=self._other_arm_collision_radius_padding,
                id_prefix=f"other_arm_sweep_{sample_index:02d}",
            )
            collision_objects.extend(sample_objects)
            proxy_links.update(sample_proxy_links)
        self._other_arm_proxy_links = proxy_links
        if not collision_objects:
            other_arm_collision_links = sorted(
                {
                    link_name
                    for link_name in self._collision_sphere_links
                    if self._is_other_arm_link(link_name)
                    and link_name not in self._other_arm_config.proxy_excluded_links
                }
            )
            raise RuntimeError(
                "other-arm collision object injection produced 0 proxy spheres "
                f"for {self._other_arm_side} arm using {self._robot_xrdf}; "
                f"source={proxy_source}; "
                f"available_other_arm_links={other_arm_collision_links or 'none'}"
            )
        self.get_logger().info(
            f"Injected {len(collision_objects)} other-arm collision spheres "
            f"for {self._other_arm_side} arm from {len(sample_maps)} {proxy_source} "
            f"sample(s); proxy_links={sorted(proxy_links)}; "
            f"excluded_proxy_links={sorted(self._other_arm_config.proxy_excluded_links)}; "
            f"padding={self._other_arm_collision_radius_padding:.3f}m; "
            f"model={self._robot_xrdf}.",
            throttle_duration_sec=5.0,
        )
        return collision_objects

    def _log_peer_plan_debug_summary(self) -> None:
        peer_plan = self._peer_plan_state
        if not isinstance(peer_plan, dict):
            self.get_logger().info(
                f"No peer plan available while planning {self._planning_arm_side} arm."
            )
            return

        joint_names = list(peer_plan.get("joint_names") or [])
        points = list(peer_plan.get("points") or [])
        published_ns = int(peer_plan.get("published_ns") or 0)
        final_time = float(peer_plan.get("final_time") or 0.0)
        if not joint_names or not points or published_ns <= 0:
            self.get_logger().info(
                f"Peer plan for {self._other_arm_side} arm is empty or invalid."
            )
            return

        now_ns = int(self.get_clock().now().nanoseconds)
        elapsed = max(0.0, (now_ns - published_ns) * 1e-9)
        sample_times = self._peer_sweep_sample_times(elapsed, final_time)
        first_positions = list(points[0].get("positions") or [])
        last_positions = list(points[-1].get("positions") or [])
        self.get_logger().info(
            f"Peer plan debug for {self._other_arm_side} arm before planning "
            f"{self._planning_arm_side}: "
            f"points={len(points)}, "
            f"elapsed={elapsed:.3f}s, final_time={final_time:.3f}s, "
            f"sample_times={[round(value, 3) for value in sample_times]}"
        )
        self.get_logger().info(
            f"Peer plan endpoints for {self._other_arm_side}: "
            f"start={self._joint_position_summary(joint_names, first_positions)}, "
            f"end={self._joint_position_summary(joint_names, last_positions)}"
        )

    def _peer_sweep_position_maps(self) -> tuple[list[dict[str, float]], str]:
        current_positions = self._latest_joint_position_map()
        peer_plan = self._peer_plan_state
        if not isinstance(peer_plan, dict):
            return [current_positions], "static"

        joint_names = list(peer_plan.get("joint_names") or [])
        points = list(peer_plan.get("points") or [])
        published_ns = int(peer_plan.get("published_ns") or 0)
        final_time = float(peer_plan.get("final_time") or 0.0)
        if not joint_names or not points or published_ns <= 0:
            return [current_positions], "static-invalid-peer-plan"

        now_ns = int(self.get_clock().now().nanoseconds)
        elapsed = max(0.0, (now_ns - published_ns) * 1e-9)
        if elapsed > final_time + self._peer_plan_extra_horizon_seconds:
            self.get_logger().info(
                f"Ignoring stale peer plan for {self._other_arm_side} arm; "
                f"elapsed={elapsed:.3f}s, final_time={final_time:.3f}s, "
                f"extra_horizon={self._peer_plan_extra_horizon_seconds:.3f}s. "
                "Using static current joint-state proxy.",
                throttle_duration_sec=5.0,
            )
            return [current_positions], "static-stale-peer-plan"

        sample_times = self._peer_sweep_sample_times(elapsed, final_time)
        samples = []
        for sample_time in sample_times:
            sample = sample_peer_plan_point(points, sample_time)
            positions = list(sample.get("positions") or [])
            position_map = dict(current_positions)
            position_map.update(
                {
                    joint_name: float(position)
                    for joint_name, position in zip(
                        joint_names,
                        positions,
                        strict=False,
                    )
                }
            )
            samples.append(position_map)
        return (samples or [current_positions]), "swept-peer-plan"

    def _latest_joint_position_map(self) -> dict[str, float]:
        if self._latest_joint_state is None:
            return {}
        return {
            name: float(position)
            for name, position in zip(
                self._latest_joint_state.name,
                self._latest_joint_state.position,
                strict=False,
            )
        }

    def _joint_position_summary(
        self,
        joint_names: list[str],
        positions: list[float],
    ) -> str:
        return ", ".join(
            f"{joint_name}={float(position):.4f}"
            for joint_name, position in zip(joint_names, positions, strict=False)
        )

    def _peer_sweep_sample_times(self, elapsed: float, final_time: float) -> list[float]:
        if self._other_arm_sweep_sample_count <= 1 or final_time <= elapsed:
            return [min(elapsed, final_time)]
        count = self._other_arm_sweep_sample_count
        span = final_time - elapsed
        return [elapsed + span * index / (count - 1) for index in range(count)]

    def _build_direct_trajectory_goal(self, trajectory: Any) -> Any:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = deepcopy(trajectory.joint_trajectory)
        goal.goal_time_tolerance = self._duration_seconds_msg(
            self._direct_trajectory_goal_time_tolerance
        )
        return goal

    def _duration_seconds_msg(self, seconds: float) -> Duration:
        duration = Duration()
        duration.sec = int(seconds)
        duration.nanosec = int((seconds - duration.sec) * 1e9)
        return duration

    def _publish_gripper_command(self, closed: bool) -> None:
        joint_command = JointState()
        joint_command.header.stamp = self.get_clock().now().to_msg()
        joint_command.name = [
            self._gripper_joint,
            self._gripper_mimic_joint,
        ]
        position = (
            self._gripper_closed_position
            if closed
            else self._gripper_open_position
        )
        joint_command.position = [position, position]
        self._joint_command_publisher.publish(joint_command)

    def _build_start_state(self) -> Any:
        if self._latest_joint_state is None:
            raise RuntimeError("Cannot build MoveGroup start_state without /joint_states.")
        start_state = RobotState()
        start_state.is_diff = False
        latest = self._latest_joint_state
        filtered_joint_state = JointState()
        filtered_joint_state.header = deepcopy(latest.header)
        allowed_names = set(self._planning_start_joints)
        for name, position in zip(latest.name, latest.position, strict=False):
            if name not in allowed_names:
                continue
            filtered_joint_state.name.append(name)
            filtered_joint_state.position.append(position)
        start_state.joint_state = filtered_joint_state
        self._clamp_start_state_to_joint_limits(start_state.joint_state)
        return start_state

    def _clamp_start_state_to_joint_limits(self, joint_state: JointState) -> None:
        for index, joint_name in enumerate(joint_state.name):
            limits = JOINT_POSITION_LIMITS.get(joint_name)
            if limits is None or index >= len(joint_state.position):
                continue
            lower, upper = limits
            joint_state.position[index] = min(
                max(joint_state.position[index], lower + JOINT_LIMIT_MARGIN_RAD),
                upper - JOINT_LIMIT_MARGIN_RAD,
            )

    def _build_pose_constraints(self, step: MotionStep) -> Any:
        if step.pose is None:
            raise RuntimeError(f"Cannot build pose constraints for {step.name!r} without pose.")

        pose = step.pose
        constraints = Constraints()
        constraints.name = f"{self._planning_arm_side}_arm_grasp_pose"

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
        request_id: str,
        request_summary: str,
        request_profile: dict[str, Any],
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
                self._write_move_group_debug_artifact(
                    request_id,
                    "goal_response_error",
                    {
                        "step_name": step.name,
                        "error": str(error),
                        "profile": request_profile,
                    },
                )
                self._step_running = False
                self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")
            return
        if not goal_handle.accepted:
            self.get_logger().error(f"MoveGroup rejected pick step {step.name!r}.")
            self._write_move_group_debug_artifact(
                request_id,
                "rejected",
                {
                    "step_name": step.name,
                    "profile": request_profile,
                },
            )
            self._step_running = False
            self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")
            return

        self.get_logger().info(
            f"MoveGroup accepted request_id={request_id} pick step {step.name!r}; waiting for result. "
            f"Request: {request_summary}"
        )
        self._active_goal_handle = goal_handle
        self._active_move_step_name = step.name
        self._active_move_request_id = request_id
        self._active_move_request_summary = request_summary
        self._active_move_request_profile = request_profile
        self._active_move_started_monotonic = time.monotonic()
        self._move_group_goal_may_be_active = True
        self._write_move_group_debug_artifact(
            request_id,
            "accepted",
            {
                "step_name": step.name,
                "profile": request_profile,
            },
        )
        self._move_result_timer = self.create_timer(
            self._move_group_result_timeout,
            lambda current_step=step, current_generation=generation: self._handle_move_result_timeout(
                current_step,
                current_generation,
            ),
        )
        if (
            self._move_group_observability_interval > 0.0
            and self._move_group_observability_interval < self._move_group_result_timeout
        ):
            self._move_observability_timer = self.create_timer(
                self._move_group_observability_interval,
                self._log_active_move_wait,
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
        if self._move_observability_timer is not None:
            self._move_observability_timer.cancel()
            self._move_observability_timer = None
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
            self._write_move_group_debug_artifact(
                self._active_move_request_id,
                "result_error",
                {
                    "step_name": step.name,
                    "error": str(error),
                    "profile": self._active_move_request_profile,
                },
            )
            self._step_running = False
            self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")
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
        planned_trajectory = getattr(result, "planned_trajectory", None)
        if error_code != 1:
            self._record_plan_outcome(
                step_name=step.name,
                success=False,
                request_id=self._active_move_request_id,
                error_code=error_code,
                error_name=error_name,
                action_status=action_status,
                planning_time=planning_time,
                planned_points=planned_points,
            )
            self.get_logger().error(
                f"MoveGroup failed pick step {step.name!r}: "
                f"error={error_name} ({error_code}), action_status={action_status}, "
                f"planning_time={planning_time:.3f}s, "
                f"planned_points={planned_points}, executed_points={executed_points}, "
                f"request_id={self._active_move_request_id}. "
                f"Request: {self._active_move_request_summary}"
            )
            self._write_move_group_debug_artifact(
                self._active_move_request_id,
                "failed",
                {
                    "step_name": step.name,
                    "error_name": error_name,
                    "error_code": error_code,
                    "action_status": action_status,
                    "planning_time": planning_time,
                    "planned_points": planned_points,
                    "executed_points": executed_points,
                    "planned_final": planned_final,
                    "executed_final": executed_final,
                    "profile": self._active_move_request_profile,
                },
            )
            self._step_running = False
            self._clear_active_move_observability()
            self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")
            if error_code in RELEASE_TARGET_ON_MOVEIT_ERROR_CODES:
                self._release_latched_target("failed MoveGroup plan")
            return

        self.get_logger().info(
            f"MoveGroup planned pick step {step.name!r}: "
            f"error={error_name} ({error_code}), action_status={action_status}, "
            f"planning_time={planning_time:.3f}s, "
            f"planned_points={planned_points}, "
            f"request_id={self._active_move_request_id}."
        )
        if planned_final:
            self.get_logger().info(
                f"MoveGroup planned final joints for {step.name!r}: {planned_final}"
            )
        self._publish_display_trajectory(planned_trajectory)
        self._log_planned_trajectory_collisions(step, planned_trajectory)
        self._current_plan_state = self._planned_trajectory_state(planned_trajectory)
        self._publish_peer_plan_state()
        self._record_plan_outcome(
            step_name=step.name,
            success=True,
            request_id=self._active_move_request_id,
            error_code=error_code,
            error_name=error_name,
            action_status=action_status,
            planning_time=planning_time,
            planned_points=planned_points,
        )
        self._publish_arm_state()
        self._write_move_group_debug_artifact(
            self._active_move_request_id,
            "succeeded",
            {
                "step_name": step.name,
                "error_name": error_name,
                "error_code": error_code,
                "action_status": action_status,
                "planning_time": planning_time,
                "planned_points": planned_points,
                "executed_points": executed_points,
                "planned_final": planned_final,
                "executed_final": executed_final,
                "profile": self._active_move_request_profile,
            },
        )
        if self._plan_only:
            self._clear_active_move_observability()
            self._step_running = False
            self._set_step_status(self._active_step_command_id, self._active_step_name, "succeeded")
            return
        if planned_points <= 0 or planned_trajectory is None:
            self.get_logger().error(
                f"MoveGroup returned no planned trajectory for pick step {step.name!r}."
            )
            self._step_running = False
            self._clear_active_move_observability()
            self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")
            self._release_latched_target("empty MoveGroup plan")
            return
        self._send_direct_trajectory(step, planned_trajectory, generation)

    def _publish_display_trajectory(self, planned_trajectory: Any | None) -> None:
        if (
            planned_trajectory is None
            or DisplayTrajectory is None
            or self._display_trajectory_publisher is None
        ):
            return
        display = DisplayTrajectory()
        display.model_id = "yumi"
        try:
            display.trajectory_start = self._build_start_state()
        except RuntimeError:
            display.trajectory_start = RobotState()
        display.trajectory.append(deepcopy(planned_trajectory))
        self._display_trajectory_publisher.publish(display)

    def _planned_trajectory_state(self, planned_trajectory: Any | None) -> dict[str, Any] | None:
        return serialize_planned_trajectory(
            planned_trajectory,
            int(self.get_clock().now().nanoseconds),
        )

    def _collision_position_map(self) -> dict[str, float]:
        return merged_position_map(
            self._latest_joint_state,
            self._peer_plan_state,
            int(self.get_clock().now().nanoseconds),
            self._peer_plan_extra_horizon_seconds,
        )

    def _load_collision_model(self) -> None:
        from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
        from curobo.cuda_robot_model.cuda_robot_model import (
            CudaRobotModel,
            CudaRobotModelConfig,
        )
        from curobo.cuda_robot_model.util import load_robot_yaml
        from curobo.types.base import TensorDeviceType
        from curobo.types.file_path import ContentPath

        if not self._robot_xrdf.is_file():
            raise FileNotFoundError(f"cuMotion XRDF not found: {self._robot_xrdf}")
        if not self._robot_urdf.is_file():
            raise FileNotFoundError(f"cuMotion URDF not found: {self._robot_urdf}")

        self._collision_sphere_links = []
        self._collision_sphere_labels = []
        self._collision_ignored_link_pairs = set()

        with self._robot_xrdf.open("r", encoding="utf-8") as stream:
            xrdf: dict[str, Any] = yaml.safe_load(stream)
        collision = xrdf.get("collision") or {}
        geometry_name = collision.get("geometry")
        geometry = ((xrdf.get("geometry") or {}).get(geometry_name) or {})
        for link_name, spheres in (geometry.get("spheres") or {}).items():
            for local_index, _sphere in enumerate(spheres or []):
                self._collision_sphere_links.append(str(link_name))
                self._collision_sphere_labels.append(f"{link_name}[{local_index}]")
        for link_name, ignored_links in ((xrdf.get("self_collision") or {}).get("ignore") or {}).items():
            for ignored_link in ignored_links or []:
                self._collision_ignored_link_pairs.add(
                    frozenset((str(link_name), str(ignored_link)))
                )

        self._collision_tensor_args = TensorDeviceType()
        content_path = ContentPath(
            robot_xrdf_absolute_path=str(self._robot_xrdf),
            robot_urdf_absolute_path=str(self._robot_urdf),
        )
        robot_yaml = load_robot_yaml(content_path)
        kinematics_config = robot_yaml["robot_cfg"]["kinematics"]
        model_config = CudaRobotModelConfig.from_config(
            CudaRobotGeneratorConfig(
                **kinematics_config,
                tensor_args=self._collision_tensor_args,
            )
        )
        self._collision_model = CudaRobotModel(model_config)
        self._collision_joint_names = list(self._collision_model.joint_names)
        planning_links = sorted(
            {
                link_name
                for link_name in self._collision_sphere_links
                if self._is_planning_arm_link(link_name)
            }
        )
        other_links = sorted(
            {
                link_name
                for link_name in self._collision_sphere_links
                if self._is_other_arm_link(link_name)
            }
        )
        self.get_logger().info(
            f"Loaded collision model from {self._robot_xrdf}: "
            f"joints={self._collision_joint_names}; "
            f"planning_arm_links={planning_links}; "
            f"other_arm_links={other_links}; "
            f"spheres={len(self._collision_sphere_links)}."
        )

    def _find_sphere_collisions(
        self,
        spheres: Any,
    ) -> list[tuple[float, str, str, float, float]]:
        collisions: list[tuple[float, str, str, float, float]] = []
        active = []
        for index, sphere in enumerate(spheres):
            x, y, z, radius = [float(value) for value in sphere]
            if radius > 0.0:
                active.append(
                    (
                        index,
                        self._collision_sphere_links[index],
                        self._collision_sphere_labels[index],
                        x,
                        y,
                        z,
                        radius,
                    )
                )
        for first_offset, first in enumerate(active):
            (
                _first_index,
                first_link,
                first_label,
                first_x,
                first_y,
                first_z,
                first_radius,
            ) = first
            for second in active[first_offset + 1 :]:
                (
                    _second_index,
                    second_link,
                    second_label,
                    second_x,
                    second_y,
                    second_z,
                    second_radius,
                ) = second
                if first_link == second_link:
                    continue
                if frozenset((first_link, second_link)) in self._collision_ignored_link_pairs:
                    continue
                dx = first_x - second_x
                dy = first_y - second_y
                dz = first_z - second_z
                distance = (dx * dx + dy * dy + dz * dz) ** 0.5
                radius_sum = first_radius + second_radius
                penetration = radius_sum - distance
                if penetration > 0.001:
                    collisions.append(
                        (
                            penetration,
                            first_label,
                            second_label,
                            distance,
                            radius_sum,
                        )
                    )
        collisions.sort(reverse=True)
        return collisions

    def _trajectory_position_samples(
        self,
        points: list[Any],
    ) -> list[tuple[str, list[float]]]:
        samples: list[tuple[str, list[float]]] = []
        if not points:
            return samples
        samples.append(("waypoint[0]", [float(value) for value in points[0].positions]))
        for index in range(1, len(points)):
            previous = points[index - 1]
            current = points[index]
            previous_positions = [float(value) for value in previous.positions]
            current_positions = [float(value) for value in current.positions]
            midpoint_positions = [
                (previous_position + current_position) * 0.5
                for previous_position, current_position in zip(
                    previous_positions,
                    current_positions,
                    strict=False,
                )
            ]
            samples.append((f"midpoint[{index - 1}->{index}]", midpoint_positions))
            samples.append((f"waypoint[{index}]", current_positions))
        return samples

    def _log_planned_trajectory_collisions(
        self,
        step: MotionStep,
        planned_trajectory: Any | None,
    ) -> None:
        if (
            not self._planned_trajectory_collision_diagnostic
            or planned_trajectory is None
            or self._latest_joint_state is None
        ):
            return
        joint_trajectory = getattr(planned_trajectory, "joint_trajectory", None)
        if joint_trajectory is None:
            return
        points = list(getattr(joint_trajectory, "points", []))
        if not points:
            return
        try:
            if self._collision_model is None or self._collision_tensor_args is None:
                self._load_collision_model()
        except Exception as error:
            self.get_logger().warn(
                f"Planned trajectory collision diagnostic unavailable: {error}",
                throttle_duration_sec=5.0,
            )
            return

        import torch

        position_map = {
            name: float(position)
            for name, position in zip(
                self._latest_joint_state.name,
                self._latest_joint_state.position,
                strict=False,
            )
        }
        missing = [
            joint_name
            for joint_name in self._collision_joint_names
            if joint_name not in position_map
        ]
        if missing:
            self.get_logger().warn(
                "Skipping planned trajectory collision diagnostic; missing joint state for "
                + ", ".join(missing),
                throttle_duration_sec=5.0,
            )
            return

        trajectory_joint_names = list(getattr(joint_trajectory, "joint_names", []))
        worst = None
        worst_sample_label = ""
        worst_collision_count = 0
        colliding_samples = 0
        colliding_sample_labels: list[str] = []
        worst_sample_positions: list[float] | None = None
        for sample_label, sample_positions in self._trajectory_position_samples(points):
            for joint_name, position in zip(
                trajectory_joint_names, sample_positions, strict=False
            ):
                position_map[joint_name] = float(position)
            q = torch.tensor(
                [[position_map[joint_name] for joint_name in self._collision_joint_names]],
                device=self._collision_tensor_args.device,
                dtype=self._collision_tensor_args.dtype,
            )
            spheres = (
                self._collision_model.get_state(q)
                .link_spheres_tensor[0]
                .detach()
                .cpu()
                .numpy()
            )
            collisions = self._find_sphere_collisions(spheres)
            if not collisions:
                continue
            colliding_samples += 1
            colliding_sample_labels.append(sample_label)
            if worst is None or collisions[0][0] > worst[0]:
                worst = collisions[0]
                worst_sample_label = sample_label
                worst_collision_count = len(collisions)
                worst_sample_positions = list(sample_positions)
        if worst is None:
            self._last_planned_collision_waypoint_index = None
            self._last_planned_collision_waypoint_joint_names = []
            self._last_planned_collision_waypoint_positions = []
            self.get_logger().info(
                f"Planned trajectory collision diagnostic for {step.name!r}: "
                f"no non-ignored sphere overlaps across {len(points)} waypoints "
                f"or interpolated midpoints "
                f"(request_id={self._active_move_request_id})."
            )
            return
        penetration, first_label, second_label, distance, radius_sum = worst
        sample_joint_summary = "unknown"
        if worst_sample_positions is not None:
            self._last_planned_collision_waypoint_joint_names = list(trajectory_joint_names)
            self._last_planned_collision_waypoint_positions = list(worst_sample_positions)
            self._last_planned_collision_waypoint_index = None
            if worst_sample_label.startswith("waypoint[") and worst_sample_label.endswith("]"):
                try:
                    self._last_planned_collision_waypoint_index = int(
                        worst_sample_label[len("waypoint[") : -1]
                    )
                except ValueError:
                    self._last_planned_collision_waypoint_index = None
            sample_joint_summary = ", ".join(
                f"{joint_name}={position:.4f}"
                for joint_name, position in zip(
                    trajectory_joint_names,
                    worst_sample_positions,
                    strict=False,
                )
            )
        self.get_logger().warn(
            f"Planned trajectory collision diagnostic for {step.name!r}: "
            f"{colliding_samples} sampled states have non-ignored sphere overlaps. "
            f"Colliding samples={colliding_sample_labels}. "
            f"Worst at {worst_sample_label}: "
            f"{first_label} <-> {second_label} penetration={penetration:.4f}m "
            f"distance={distance:.4f}m radius_sum={radius_sum:.4f}m "
            f"(sample_collisions={worst_collision_count}, "
            f"sample_joints={sample_joint_summary}, "
            f"request_id={self._active_move_request_id})."
        )
        self._log_final_planned_state_collision(step, trajectory_joint_names, points[-1])

    def _log_final_planned_state_collision(
        self,
        step: MotionStep,
        trajectory_joint_names: list[str],
        final_point: Any,
    ) -> None:
        if (
            self._collision_model is None
            or self._collision_tensor_args is None
            or self._latest_joint_state is None
        ):
            return

        import torch

        position_map = {
            name: float(position)
            for name, position in zip(
                self._latest_joint_state.name,
                self._latest_joint_state.position,
                strict=False,
            )
        }
        final_positions = [float(value) for value in getattr(final_point, "positions", [])]
        for joint_name, position in zip(
            trajectory_joint_names, final_positions, strict=False
        ):
            position_map[joint_name] = float(position)

        missing = [
            joint_name
            for joint_name in self._collision_joint_names
            if joint_name not in position_map
        ]
        if missing:
            self.get_logger().warn(
                "Skipping final planned state collision diagnostic; missing joint state for "
                + ", ".join(missing),
                throttle_duration_sec=5.0,
            )
            return

        q = torch.tensor(
            [[position_map[joint_name] for joint_name in self._collision_joint_names]],
            device=self._collision_tensor_args.device,
            dtype=self._collision_tensor_args.dtype,
        )
        spheres = (
            self._collision_model.get_state(q)
            .link_spheres_tensor[0]
            .detach()
            .cpu()
            .numpy()
        )
        collisions = self._find_sphere_collisions(spheres)
        final_joint_summary = ", ".join(
            f"{joint_name}={position:.4f}"
            for joint_name, position in zip(
                trajectory_joint_names,
                final_positions,
                strict=False,
            )
        )
        if not collisions:
            self.get_logger().info(
                f"Final planned state collision diagnostic for {step.name!r}: "
                f"terminal waypoint is collision-free "
                f"(sample_joints={final_joint_summary}, "
                f"request_id={self._active_move_request_id})."
            )
            return

        penetration, first_label, second_label, distance, radius_sum = collisions[0]
        self.get_logger().warn(
            f"Final planned state collision diagnostic for {step.name!r}: "
            f"terminal waypoint has {len(collisions)} non-ignored sphere overlaps. "
            f"Worst: {first_label} <-> {second_label} penetration={penetration:.4f}m "
            f"distance={distance:.4f}m radius_sum={radius_sum:.4f}m "
            f"(sample_joints={final_joint_summary}, "
            f"request_id={self._active_move_request_id})."
        )

    def _send_direct_trajectory(
        self,
        step: MotionStep,
        planned_trajectory: Any,
        generation: int,
    ) -> None:
        if self._direct_trajectory_client is None:
            self.get_logger().error(
                f"Cannot execute {step.name!r}; direct trajectory action support is unavailable."
            )
            self._step_running = False
            self._clear_active_move_observability()
            return
        if not self._direct_trajectory_client.server_is_ready():
            self.get_logger().info("Waiting for direct trajectory action server...")
            if not self._direct_trajectory_client.wait_for_server(
                timeout_sec=MOVE_GROUP_SERVER_WAIT_SECONDS
            ):
                self.get_logger().error(
                    "Direct trajectory action server is not available; cannot execute "
                    f"pick step {step.name!r}."
                )
                self._step_running = False
                self._clear_active_move_observability()
                return

        goal = self._build_direct_trajectory_goal(planned_trajectory)
        point_count = len(goal.trajectory.points)
        final_time = 0.0
        if goal.trajectory.points:
            final = goal.trajectory.points[-1].time_from_start
            final_time = float(final.sec) + float(final.nanosec) * 1e-9
        self.get_logger().info(
            f"Sending direct trajectory for pick step {step.name!r}: "
            f"points={point_count}, duration={final_time:.3f}s, "
            f"request_id={self._active_move_request_id}."
        )
        self._move_group_goal_may_be_active = True
        self._publish_arm_state()
        self._active_move_step_name = f"{step.name} direct execution"
        self._active_move_started_monotonic = time.monotonic()
        timeout = self._direct_trajectory_result_timeout
        if timeout > 0.0:
            self._move_result_timer = self.create_timer(
                timeout,
                lambda current_step=step, current_generation=generation: self._handle_direct_trajectory_timeout(
                    current_step,
                    current_generation,
                ),
            )
        if (
            self._move_group_observability_interval > 0.0
            and timeout > self._move_group_observability_interval
        ):
            self._move_observability_timer = self.create_timer(
                self._move_group_observability_interval,
                self._log_active_move_wait,
            )
        send_future = self._direct_trajectory_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda future, current_step=step, current_generation=generation: self._handle_direct_trajectory_goal_response(
                future,
                current_step,
                current_generation,
            )
        )

    def _handle_direct_trajectory_goal_response(
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
            self.get_logger().error(
                f"Direct trajectory goal response failed for pick step {step.name!r}: {error}"
            )
            self._step_running = False
            self._clear_direct_trajectory_observability()
            self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")
            return
        if not goal_handle.accepted:
            self.get_logger().error(
                f"Direct trajectory action rejected pick step {step.name!r}."
            )
            self._step_running = False
            self._clear_direct_trajectory_observability()
            self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")
            return
        self._active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, current_step=step, current_generation=generation: self._handle_direct_trajectory_result(
                result,
                current_step,
                current_generation,
            )
        )

    def _handle_direct_trajectory_result(
        self,
        future: Any,
        step: MotionStep,
        generation: int,
    ) -> None:
        if generation != self._pick_generation:
            return
        try:
            action_result = future.result()
            result = action_result.result
        except Exception as error:
            self.get_logger().error(
                f"Direct trajectory result failed for pick step {step.name!r}: {error}"
            )
            self._step_running = False
            self._clear_direct_trajectory_observability()
            self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")
            return
        status = getattr(action_result, "status", None)
        error_code = int(getattr(result, "error_code", -1))
        error_string = str(getattr(result, "error_string", ""))
        if error_code != 0:
            self.get_logger().error(
                f"Direct trajectory failed pick step {step.name!r}: "
                f"error_code={error_code}, action_status={status}, "
                f"error={error_string}, request_id={self._active_move_request_id}."
            )
            self._step_running = False
            self._clear_direct_trajectory_observability()
            self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")
            return
        self.get_logger().info(
            f"Direct trajectory completed pick step {step.name!r}: "
            f"action_status={status}, request_id={self._active_move_request_id}."
        )
        self._log_actual_vs_planned_collision_waypoint(step)
        self._write_move_group_debug_artifact(
            self._active_move_request_id,
            "executed",
            {
                "step_name": step.name,
                "action_status": status,
                "error_code": error_code,
            },
        )
        self._current_plan_state = None
        self._publish_arm_state()
        self._clear_direct_trajectory_observability()
        self._step_running = False
        self._set_step_status(self._active_step_command_id, self._active_step_name, "succeeded")

    def _handle_direct_trajectory_timeout(
        self,
        step: MotionStep,
        generation: int,
    ) -> None:
        if generation != self._pick_generation:
            return
        elapsed = (
            time.monotonic() - self._active_move_started_monotonic
            if self._active_move_started_monotonic > 0.0
            else self._direct_trajectory_result_timeout
        )
        self._pick_generation += 1
        self._step_running = False
        self._active_goal_handle = None
        self.get_logger().error(
            f"Direct trajectory did not return a result for pick step {step.name!r} "
            f"within {self._direct_trajectory_result_timeout:.1f}s "
            f"(elapsed={elapsed:.1f}s). request_id={self._active_move_request_id}."
        )
        self._write_move_group_debug_artifact(
            self._active_move_request_id,
            "direct_timeout",
            {
                "step_name": step.name,
                "elapsed": elapsed,
                "timeout_seconds": self._direct_trajectory_result_timeout,
                "profile": self._active_move_request_profile,
            },
        )
        self._clear_direct_trajectory_observability()
        self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")

    def _log_actual_vs_planned_collision_waypoint(self, step: MotionStep) -> None:
        if (
            self._latest_joint_state is None
            or self._last_planned_collision_waypoint_index is None
            or not self._last_planned_collision_waypoint_joint_names
            or not self._last_planned_collision_waypoint_positions
        ):
            return
        actual_positions = {
            name: float(position)
            for name, position in zip(
                self._latest_joint_state.name,
                self._latest_joint_state.position,
                strict=False,
            )
        }
        comparisons = []
        for joint_name, planned_position in zip(
            self._last_planned_collision_waypoint_joint_names,
            self._last_planned_collision_waypoint_positions,
            strict=False,
        ):
            if joint_name not in actual_positions:
                continue
            actual_position = actual_positions[joint_name]
            comparisons.append(
                f"{joint_name}: planned={planned_position:.4f}, "
                f"actual={actual_position:.4f}, "
                f"delta={actual_position - planned_position:.4f}"
            )
        if comparisons:
            self.get_logger().info(
                f"Actual vs worst planned collision waypoint for {step.name!r} "
                f"(waypoint={self._last_planned_collision_waypoint_index}, "
                f"request_id={self._active_move_request_id}): "
                + "; ".join(comparisons)
            )

    def _clear_direct_trajectory_observability(self) -> None:
        self._move_group_goal_may_be_active = False
        if self._move_result_timer is not None:
            self._move_result_timer.cancel()
            self._move_result_timer = None
        if self._move_observability_timer is not None:
            self._move_observability_timer.cancel()
            self._move_observability_timer = None
        self._active_goal_handle = None
        self._current_plan_state = None
        self._publish_arm_state()
        self._clear_active_move_observability()

    def _handle_move_result_timeout(self, step: MotionStep, generation: int) -> None:
        if self._move_result_timer is not None:
            self._move_result_timer.cancel()
            self._move_result_timer = None
        if self._move_observability_timer is not None:
            self._move_observability_timer.cancel()
            self._move_observability_timer = None
        if generation != self._pick_generation:
            return
        elapsed = (
            time.monotonic() - self._active_move_started_monotonic
            if self._active_move_started_monotonic > 0.0
            else self._move_group_result_timeout
        )
        self._pick_generation += 1
        self._step_running = False
        self._active_goal_handle = None
        self.get_logger().error(
            f"MoveGroup did not return a result for pick step {step.name!r} within "
            f"{self._move_group_result_timeout:.1f}s "
            f"(elapsed={elapsed:.1f}s). The goal may still be active "
            "inside MoveGroup; not canceling it because preempt can crash this setup. "
            f"Restart move_group before retrying. request_id={self._active_move_request_id}. "
            f"Request: {self._active_move_request_summary}"
        )
        self._write_move_group_debug_artifact(
            self._active_move_request_id,
            "timeout",
            {
                "step_name": step.name,
                "elapsed": elapsed,
                "timeout_seconds": self._move_group_result_timeout,
                "profile": self._active_move_request_profile,
            },
        )
        self._move_group_goal_may_be_active = False
        self._current_plan_state = None
        self._publish_arm_state()
        self._clear_active_move_observability()
        self._set_step_status(self._active_step_command_id, self._active_step_name, "failed")

    def _log_active_move_wait(self) -> None:
        if not self._move_group_goal_may_be_active or not self._active_move_step_name:
            return
        elapsed = time.monotonic() - self._active_move_started_monotonic
        self.get_logger().warning(
            f"Still waiting on MoveGroup result for request_id={self._active_move_request_id} "
            f"pick step {self._active_move_step_name!r} after {elapsed:.1f}s. "
            f"Request: {self._active_move_request_summary}"
        )
        self._write_move_group_debug_artifact(
            self._active_move_request_id,
            "waiting",
            {
                "step_name": self._active_move_step_name,
                "elapsed": elapsed,
                "profile": self._active_move_request_profile,
            },
        )

    def _clear_active_move_observability(self) -> None:
        self._active_move_step_name = ""
        self._active_move_request_id = ""
        self._active_move_request_summary = ""
        self._active_move_request_profile = {}
        self._active_move_started_monotonic = 0.0

    def _build_move_group_observability(
        self,
        step: MotionStep,
        goal: Any,
    ) -> tuple[str, dict[str, Any]]:
        request = goal.request
        pose = step.pose.pose if step.pose is not None else None
        self._move_request_serial += 1
        request_id = f"move-{self._move_request_serial:05d}"
        pose_data: dict[str, Any] = {}
        if pose is not None:
            pose_data = {
                "frame": step.pose.header.frame_id,
                "position_xyz": [
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                ],
                "orientation_xyzw": [
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ],
                "position_norm": (
                    pose.position.x ** 2 + pose.position.y ** 2 + pose.position.z ** 2
                ) ** 0.5,
            }
        start_state_summary = self._joint_state_summary(
            request.start_state.joint_state,
            self._planning_joint_order,
        )
        orientation_data: dict[str, Any] = {"enabled": False}
        if request.goal_constraints:
            orientation_constraints = getattr(
                request.goal_constraints[0],
                "orientation_constraints",
                [],
            )
            if orientation_constraints:
                constraint = orientation_constraints[0]
                orientation_data = {
                    "enabled": True,
                    "orientation_xyzw": [
                        float(constraint.orientation.x),
                        float(constraint.orientation.y),
                        float(constraint.orientation.z),
                        float(constraint.orientation.w),
                    ],
                    "tolerance_xyz": [
                        float(constraint.absolute_x_axis_tolerance),
                        float(constraint.absolute_y_axis_tolerance),
                        float(constraint.absolute_z_axis_tolerance),
                    ],
                    "weight": float(constraint.weight),
                }
        pose_position = pose_data.get("position_xyz", [0.0, 0.0, 0.0])
        pose_orientation = pose_data.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])
        summary = (
            f"pipeline={request.pipeline_id} planner={request.planner_id} "
            f"group={request.group_name} attempts={request.num_planning_attempts} "
            f"allowed_planning_time={request.allowed_planning_time:.2f}s "
            f"vel_scale={request.max_velocity_scaling_factor:.3f} "
            f"acc_scale={request.max_acceleration_scaling_factor:.3f} "
            f"position_tolerance={self._position_tolerance:.3f} "
            f"target=({pose_position[0]:.3f}, {pose_position[1]:.3f}, {pose_position[2]:.3f}) "
            f"quat=({pose_orientation[0]:.3f}, {pose_orientation[1]:.3f}, "
            f"{pose_orientation[2]:.3f}, {pose_orientation[3]:.3f}) "
            f"frame={pose_data.get('frame', '?')}; "
            f"orientation_constraint={'on' if orientation_data['enabled'] else 'off'}; "
            f"start_state={start_state_summary}"
        )
        profile = {
            "request_id": request_id,
            "step_name": step.name,
            "timestamp_ns": int(self.get_clock().now().nanoseconds),
            "pipeline_id": request.pipeline_id,
            "planner_id": request.planner_id,
            "group_name": request.group_name,
            "num_planning_attempts": int(request.num_planning_attempts),
            "allowed_planning_time": float(request.allowed_planning_time),
            "max_velocity_scaling_factor": float(request.max_velocity_scaling_factor),
            "max_acceleration_scaling_factor": float(
                request.max_acceleration_scaling_factor
            ),
            "position_tolerance": float(self._position_tolerance),
            "force_top_down": bool(step.force_top_down),
            "pose": pose_data,
            "orientation_constraint": orientation_data,
            "start_state": self._joint_state_dict(
                request.start_state.joint_state,
                self._planning_joint_order,
            ),
            "summary": summary,
        }
        return request_id, profile

    def _joint_state_summary(
        self,
        joint_state: JointState,
        joint_names: tuple[str, ...],
    ) -> str:
        positions_by_name = {
            joint_name: joint_state.position[index]
            for index, joint_name in enumerate(joint_state.name)
            if index < len(joint_state.position)
        }
        return ", ".join(
            f"{joint_name}={positions_by_name[joint_name]:.4f}"
            for joint_name in joint_names
            if joint_name in positions_by_name
        )

    def _joint_state_dict(
        self,
        joint_state: JointState,
        joint_names: tuple[str, ...],
    ) -> dict[str, float]:
        positions_by_name = {
            joint_name: joint_state.position[index]
            for index, joint_name in enumerate(joint_state.name)
            if index < len(joint_state.position)
        }
        return {
            joint_name: float(positions_by_name[joint_name])
            for joint_name in joint_names
            if joint_name in positions_by_name
        }

    def _write_move_group_debug_artifact(
        self,
        request_id: str,
        phase: str,
        payload: dict[str, Any],
    ) -> None:
        if not request_id:
            return
        artifact_path = self._move_group_debug_dump_dir / f"{request_id}_{phase}.json"
        document = {
            "request_id": request_id,
            "phase": phase,
            "wall_time_monotonic": time.monotonic(),
            "payload": payload,
        }
        try:
            artifact_path.write_text(
                json.dumps(document, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as error:
            self.get_logger().warning(
                f"Failed to write MoveGroup debug artifact {artifact_path}: {error}"
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
