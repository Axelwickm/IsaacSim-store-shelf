#!/usr/bin/env python3

from dataclasses import dataclass

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


@dataclass(frozen=True)
class ExternalDependency:
    package_name: str
    purpose: str


REQUIRED_EXTERNAL_PACKAGES = (
    ExternalDependency("moveit_core", "MoveIt core libraries"),
    ExternalDependency("moveit_ros_move_group", "MoveIt move_group runtime"),
    ExternalDependency("isaac_ros_cumotion", "cuMotion planner runtime"),
)


class MotionPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_planner")
        self.declare_parameter("planning_group", "yumi_arm")
        self.declare_parameter("pipeline_id", "isaac_ros_cumotion")
        self.declare_parameter("planner_id", "cuMotion")
        self.declare_parameter("target_pose_topic", "/motion/target_pose")

        self._validate_external_dependencies()

        planning_group = str(self.get_parameter("planning_group").value)
        pipeline_id = str(self.get_parameter("pipeline_id").value)
        planner_id = str(self.get_parameter("planner_id").value)
        target_pose_topic = str(self.get_parameter("target_pose_topic").value)

        self._target_pose_subscription = self.create_subscription(
            PoseStamped,
            target_pose_topic,
            self._handle_target_pose,
            10,
        )

        self.get_logger().info(
            "Motion planner scaffold ready "
            f"(group={planning_group}, pipeline={pipeline_id}, planner={planner_id})"
        )
        self.get_logger().info(
            f"Listening for target poses on {target_pose_topic}. "
            "Next step is wiring this node to MoveIt's action or planning interface."
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

    def _handle_target_pose(self, message: PoseStamped) -> None:
        self.get_logger().info(
            "Received target pose in frame "
            f"{message.header.frame_id!r} at "
            f"({message.pose.position.x:.3f}, "
            f"{message.pose.position.y:.3f}, "
            f"{message.pose.position.z:.3f})."
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
