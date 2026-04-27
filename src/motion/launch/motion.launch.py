from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    planning_group = LaunchConfiguration("planning_group")
    pipeline_id = LaunchConfiguration("pipeline_id")
    planner_id = LaunchConfiguration("planner_id")
    target_pose_topic = LaunchConfiguration("target_pose_topic")
    plan_only = LaunchConfiguration("plan_only")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "planning_group",
                default_value="yumi_arm",
                description="MoveIt planning group to use.",
            ),
            DeclareLaunchArgument(
                "pipeline_id",
                default_value="isaac_ros_cumotion",
                description="MoveIt planning pipeline identifier.",
            ),
            DeclareLaunchArgument(
                "planner_id",
                default_value="cuMotion",
                description="Planner identifier for MoveIt requests.",
            ),
            DeclareLaunchArgument(
                "target_pose_topic",
                default_value="/motion/target_pose",
                description="PoseStamped topic to consume motion targets from.",
            ),
            DeclareLaunchArgument(
                "plan_only",
                default_value="false",
                description="Whether MoveIt requests should plan only without execution.",
            ),
            Node(
                package="motion",
                executable="planner",
                name="motion_planner",
                output="screen",
                parameters=[
                    {
                        "planning_group": planning_group,
                        "pipeline_id": pipeline_id,
                        "planner_id": planner_id,
                        "target_pose_topic": target_pose_topic,
                        "plan_only": plan_only,
                    }
                ],
            ),
            Node(
                package="motion",
                executable="trajectory_controller",
                name="right_arm_trajectory_controller",
                output="screen",
            ),
        ]
    )
