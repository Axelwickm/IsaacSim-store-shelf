#!/usr/bin/env python3

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

try:
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import (
        BoundingVolume,
        Constraints,
        DisplayTrajectory,
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
    Duration = None
    FollowJointTrajectory = None
    DisplayTrajectory = None
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

RELEASE_TARGET_ON_MOVEIT_ERROR_CODES = {
    99999,  # Generic planning failure from MoveIt/OMPL.
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
JOINT_POSITION_LIMITS = {
    "yumi_joint_1_r": (-2.84, 2.84),
    "yumi_joint_2_r": (-2.4, 0.65),
    "yumi_joint_7_r": (-2.84, 2.84),
    "yumi_joint_3_r": (-2.0, 1.29),
    "yumi_joint_4_r": (-4.9, 4.9),
    "yumi_joint_5_r": (-1.43, 2.3),
    "yumi_joint_6_r": (-3.89, 3.89),
}
JOINT_LIMIT_MARGIN_RAD = 1e-3
RIGHT_ARM_JOINT_ORDER = (
    "yumi_joint_1_r",
    "yumi_joint_2_r",
    "yumi_joint_7_r",
    "yumi_joint_3_r",
    "yumi_joint_4_r",
    "yumi_joint_5_r",
    "yumi_joint_6_r",
)


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
        self.declare_parameter(
            "direct_trajectory_action",
            "right_arm_controller/follow_joint_trajectory",
        )
        self.declare_parameter("direct_trajectory_result_timeout", 30.0)
        self.declare_parameter("direct_trajectory_goal_time_tolerance", 30.0)
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("right_gripper_joint", "gripper_r_joint")
        self.declare_parameter("right_gripper_mimic_joint", "gripper_r_joint_m")
        self.declare_parameter("right_gripper_open_position", 0.025)
        self.declare_parameter("right_gripper_closed_position", 0.0)
        self.declare_parameter("gripper_command_settle_seconds", 0.0)
        self.declare_parameter("target_pose_topic", "/motion/target_pose")
        self.declare_parameter("selected_item_point_topic", "/vision/selected_item_point")
        self.declare_parameter(
            "selected_item_moveit_point_topic",
            "/vision/selected_item_moveit_point",
        )
        self.declare_parameter("latched_target_point_topic", "/motion/latched_target_point")
        self.declare_parameter("clear_latched_target_topic", "/motion/clear_latched_target")
        self.declare_parameter(
            "display_trajectory_topic",
            "/moveit/display_planned_path",
        )
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
            [0.9, 0.9, 3.14159],
        )
        self.declare_parameter("release_top_down_orientation_weight", 0.5)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("move_group_result_timeout", 30.0)
        self.declare_parameter("move_group_observability_interval", 5.0)
        self.declare_parameter("move_group_debug_dump_dir", "/tmp/motion_planner_debug")
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
        direct_trajectory_action = str(
            self.get_parameter("direct_trajectory_action").value
        )
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
        display_trajectory_topic = str(
            self.get_parameter("display_trajectory_topic").value
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
        self._auto_execute_pick = bool(self.get_parameter("auto_execute_pick").value)
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
            f"pipeline={pipeline_id}, planner={planner_id}, "
            f"direct_execute={not self._plan_only})"
        )
        self.get_logger().info(
            f"Listening for target poses on {target_pose_topic}."
        )
        self.get_logger().info(
            f"Listening for selected item points on {selected_item_point_topic}; "
            f"planning points on {selected_item_moveit_point_topic}; "
            f"first world point is latched and republished on {latched_target_point_topic}. "
            f"Publish std_msgs/Empty on {clear_latched_target_topic} to latch a new point."
        )
        self.get_logger().info(
            f"MoveGroup debug artifacts will be written under "
            f"{self._move_group_debug_dump_dir}"
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

    def _release_latched_target(self, reason: str) -> None:
        if self._latched_target_point is None:
            return

        point = self._latched_target_point.point
        self._latched_target_point = None
        self.get_logger().info(
            f"Released {reason} target at "
            f"({point.x:.3f}, {point.y:.3f}, {point.z:.3f}); "
            "waiting to latch the next selected item point."
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
        if self._move_observability_timer is not None:
            self._move_observability_timer.cancel()
            self._move_observability_timer = None
        if self._active_goal_handle is not None:
            self.get_logger().info(
                f"Stopped local pick sequence after {reason}; not canceling active "
                "action goal because preempt can crash this setup."
            )
            self._active_goal_handle = None
        self._active_move_step_name = ""
        self._active_move_request_id = ""
        self._active_move_request_summary = ""
        self._active_move_request_profile = {}
        self._active_move_started_monotonic = 0.0

    def _execute_next_pick_step(self) -> None:
        if self._pick_sequence_index >= len(self._pick_sequence):
            self._pick_in_progress = False
            self.get_logger().info("Pick sequence completed.")
            self._release_latched_target("completed pick")
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
            if self._gripper_command_settle_seconds <= 0.0:
                self._execute_next_pick_step()
            else:
                self._step_timer = self.create_timer(
                    self._gripper_command_settle_seconds,
                    self._execute_next_pick_step_once,
                )
            return

        if step.pose is None:
            self.get_logger().error(f"Pick step {step.name!r} has no target pose.")
            self._fail_pick_sequence()
            return

        if not MOVEIT_ACTIONS_AVAILABLE or self._move_group_client is None:
            self.get_logger().warning(
                f"Cannot execute {step.name!r}; MoveGroup action support is unavailable."
            )
            self._fail_pick_sequence()
            return
        if self._move_group_goal_may_be_active:
            self.get_logger().error(
                "Not sending a new plan because a previous planning or execution goal "
                "may still be active. Restart the stack before retrying."
            )
            self._fail_pick_sequence()
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
                self._retry_current_pick_step()
                return

        goal = self._build_move_group_goal(step)
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

    def _execute_next_pick_step_once(self) -> None:
        if self._step_timer is not None:
            self._step_timer.cancel()
            self._step_timer = None
        self._execute_next_pick_step()

    def _retry_current_pick_step(self) -> None:
        self._pick_sequence_index = max(self._pick_sequence_index - 1, 0)
        self._step_timer = self.create_timer(
            MOVE_GROUP_SERVER_RETRY_SECONDS,
            self._execute_next_pick_step_once,
        )

    def _fail_pick_sequence(self) -> None:
        self._pick_in_progress = False

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
        request.group_name = self._right_arm_group
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
        return options

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
                self._pick_in_progress = False
                # self._release_latched_target("failed MoveGroup goal response")
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
            self._pick_in_progress = False
            # self._release_latched_target("rejected MoveGroup goal")
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
            self._pick_in_progress = False
            # self._release_latched_target("failed MoveGroup result")
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
            self._pick_in_progress = False
            self._clear_active_move_observability()
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
            self._execute_next_pick_step()
            return
        if planned_points <= 0 or planned_trajectory is None:
            self.get_logger().error(
                f"MoveGroup returned no planned trajectory for pick step {step.name!r}."
            )
            self._pick_in_progress = False
            self._clear_active_move_observability()
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
            self._pick_in_progress = False
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
                self._pick_in_progress = False
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
            self._pick_in_progress = False
            self._clear_direct_trajectory_observability()
            return
        if not goal_handle.accepted:
            self.get_logger().error(
                f"Direct trajectory action rejected pick step {step.name!r}."
            )
            self._pick_in_progress = False
            self._clear_direct_trajectory_observability()
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
            self._pick_in_progress = False
            self._clear_direct_trajectory_observability()
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
            self._pick_in_progress = False
            self._clear_direct_trajectory_observability()
            return
        self.get_logger().info(
            f"Direct trajectory completed pick step {step.name!r}: "
            f"action_status={status}, request_id={self._active_move_request_id}."
        )
        self._write_move_group_debug_artifact(
            self._active_move_request_id,
            "executed",
            {
                "step_name": step.name,
                "action_status": status,
                "error_code": error_code,
            },
        )
        self._clear_direct_trajectory_observability()
        self._execute_next_pick_step()

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
        self._pick_in_progress = False
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

    def _clear_direct_trajectory_observability(self) -> None:
        self._move_group_goal_may_be_active = False
        if self._move_result_timer is not None:
            self._move_result_timer.cancel()
            self._move_result_timer = None
        if self._move_observability_timer is not None:
            self._move_observability_timer.cancel()
            self._move_observability_timer = None
        self._active_goal_handle = None
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
        self._pick_in_progress = False
        self._active_goal_handle = None
        # self._release_latched_target("timed-out MoveGroup")
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
        self._clear_active_move_observability()

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
            RIGHT_ARM_JOINT_ORDER,
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
                RIGHT_ARM_JOINT_ORDER,
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
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
