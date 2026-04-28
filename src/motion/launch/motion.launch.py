from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    planning_group = LaunchConfiguration("planning_group")
    pipeline_id = LaunchConfiguration("pipeline_id")
    planner_id = LaunchConfiguration("planner_id")
    target_pose_topic = LaunchConfiguration("target_pose_topic")
    plan_only = LaunchConfiguration("plan_only")
    move_group_result_timeout = LaunchConfiguration("move_group_result_timeout")
    publish_cumotion_spheres = LaunchConfiguration("publish_cumotion_spheres")
    cumotion_robot_xrdf = LaunchConfiguration("cumotion_robot_xrdf")
    cumotion_urdf_path = LaunchConfiguration("cumotion_urdf_path")

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
            DeclareLaunchArgument(
                "move_group_result_timeout",
                default_value="120.0",
                description="Seconds to wait for a MoveGroup action result, including execution.",
            ),
            DeclareLaunchArgument(
                "publish_cumotion_spheres",
                default_value="true",
                description="Publish cuMotion robot collision spheres for RViz debugging.",
            ),
            DeclareLaunchArgument(
                "cumotion_robot_xrdf",
                default_value="/workspace/usd/robot/yumi_isaacsim.xrdf",
                description="XRDF used to compute cuMotion robot collision spheres.",
            ),
            DeclareLaunchArgument(
                "cumotion_urdf_path",
                default_value="/workspace/usd/robot/yumi_isaacsim.urdf",
                description="URDF used to compute cuMotion robot collision spheres.",
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
                        "move_group_result_timeout": move_group_result_timeout,
                    }
                ],
            ),
            Node(
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            pipeline_id,
                            "' == 'isaac_ros_cumotion' and '",
                            publish_cumotion_spheres,
                            "' == 'true'",
                        ]
                    )
                ),
                package="motion",
                executable="cumotion_sphere_publisher",
                name="cumotion_sphere_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_xrdf": cumotion_robot_xrdf,
                        "urdf_path": cumotion_urdf_path,
                        "joint_states_topic": "/joint_states",
                        "marker_topic": "/cumotion/robot_spheres",
                        "frame_id": "yumi_body",
                    }
                ],
            ),
        ]
    )
