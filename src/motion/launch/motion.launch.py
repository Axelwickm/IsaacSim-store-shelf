from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    planning_group = LaunchConfiguration("planning_group")
    launch_right_planner = LaunchConfiguration("launch_right_planner")
    launch_left_planner = LaunchConfiguration("launch_left_planner")
    pipeline_id = LaunchConfiguration("pipeline_id")
    planner_id = LaunchConfiguration("planner_id")
    plan_only = LaunchConfiguration("plan_only")
    move_group_result_timeout = LaunchConfiguration("move_group_result_timeout")
    direct_trajectory_result_timeout = LaunchConfiguration(
        "direct_trajectory_result_timeout"
    )
    direct_trajectory_goal_time_tolerance = LaunchConfiguration(
        "direct_trajectory_goal_time_tolerance"
    )
    arm_state_topic = LaunchConfiguration("arm_state_topic")
    coordinator_state_topic = LaunchConfiguration("coordinator_state_topic")
    reset_topic = LaunchConfiguration("reset_topic")
    selected_candidate_topic = LaunchConfiguration("selected_candidate_topic")
    publish_cumotion_spheres = LaunchConfiguration("publish_cumotion_spheres")
    right_cumotion_robot_xrdf = LaunchConfiguration("right_cumotion_robot_xrdf")
    right_cumotion_urdf_path = LaunchConfiguration("right_cumotion_urdf_path")
    left_cumotion_robot_xrdf = LaunchConfiguration("left_cumotion_robot_xrdf")
    left_cumotion_urdf_path = LaunchConfiguration("left_cumotion_urdf_path")
    collision_robot_xrdf = LaunchConfiguration("collision_robot_xrdf")
    collision_robot_urdf_path = LaunchConfiguration("collision_robot_urdf_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "planning_group",
                default_value="yumi_arm",
                description="MoveIt planning group to use.",
            ),
            DeclareLaunchArgument(
                "launch_right_planner",
                default_value="true",
                description="Whether to launch the right-arm planner node.",
            ),
            DeclareLaunchArgument(
                "launch_left_planner",
                default_value="true",
                description="Whether to launch the left-arm planner node.",
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
                "plan_only",
                default_value="false",
                description=(
                    "Plan with MoveIt/cuMotion but skip direct trajectory execution."
                ),
            ),
            DeclareLaunchArgument(
                "move_group_result_timeout",
                default_value="30.0",
                description="Seconds to wait for a MoveGroup plan-only action result.",
            ),
            DeclareLaunchArgument(
                "direct_trajectory_result_timeout",
                default_value="30.0",
                description="Seconds to wait for the direct Isaac trajectory action result.",
            ),
            DeclareLaunchArgument(
                "direct_trajectory_goal_time_tolerance",
                default_value="30.0",
                description="Goal time tolerance sent to the direct trajectory executor.",
            ),
            DeclareLaunchArgument(
                "arm_state_topic",
                default_value="/motion/arm_state",
                description="Topic used by arm planners to publish local coordination state.",
            ),
            DeclareLaunchArgument(
                "coordinator_state_topic",
                default_value="/motion/coordinator_state",
                description="Topic used by the coordinator to publish arbitration state.",
            ),
            DeclareLaunchArgument(
                "selected_candidate_topic",
                default_value="/vision/selected_candidate",
                description="Atomic vision candidate topic consumed by the coordinator.",
            ),
            DeclareLaunchArgument(
                "reset_topic",
                default_value="/motion/reset",
                description="Episode-reset topic used to clear coordinator and planner state.",
            ),
            DeclareLaunchArgument(
                "publish_cumotion_spheres",
                default_value="true",
                description="Publish cuMotion robot collision spheres for RViz debugging.",
            ),
            DeclareLaunchArgument(
                "right_cumotion_robot_xrdf",
                default_value="/workspace/usd/robot/yumi_isaacsim_right_arm.xrdf",
                description="Right-arm XRDF used by the standalone cuMotion planner node.",
            ),
            DeclareLaunchArgument(
                "right_cumotion_urdf_path",
                default_value="/workspace/usd/robot/yumi_isaacsim_right_arm.urdf",
                description="Right-arm URDF used by the standalone cuMotion planner node.",
            ),
            DeclareLaunchArgument(
                "left_cumotion_robot_xrdf",
                default_value="/workspace/usd/robot/yumi_isaacsim_left_arm.xrdf",
                description="Left-arm XRDF used by the standalone cuMotion planner node.",
            ),
            DeclareLaunchArgument(
                "left_cumotion_urdf_path",
                default_value="/workspace/usd/robot/yumi_isaacsim_left_arm.urdf",
                description="Left-arm URDF used by the standalone cuMotion planner node.",
            ),
            DeclareLaunchArgument(
                "collision_robot_xrdf",
                default_value="/workspace/usd/robot/yumi_isaacsim.xrdf",
                description="Full YuMi XRDF used by Python planner collision diagnostics and peer-arm proxy generation.",
            ),
            DeclareLaunchArgument(
                "collision_robot_urdf_path",
                default_value="/workspace/usd/robot/yumi_isaacsim.urdf",
                description="Full YuMi URDF used by Python planner collision diagnostics and peer-arm proxy generation.",
            ),
            Node(
                package="motion",
                executable="coordinator",
                name="motion_coordinator",
                output="screen",
                parameters=[
                    {
                        "arm_state_topic": arm_state_topic,
                        "coordinator_state_topic": coordinator_state_topic,
                        "selected_candidate_topic": selected_candidate_topic,
                        "reset_topic": reset_topic,
                    }
                ],
            ),
            Node(
                condition=IfCondition(launch_right_planner),
                package="motion",
                executable="planner",
                name="motion_planner_right",
                output="screen",
                parameters=[
                    {
                        "planning_group": planning_group,
                        "planning_arm_side": "right",
                        "pipeline_id": pipeline_id,
                        "planner_id": planner_id,
                        "move_group_action": "/moveit_right/move_action",
                        "planning_scene_service": "/moveit_right/get_planning_scene",
                        "planning_scene_topic": "/moveit_right/planning_scene",
                        "plan_only": plan_only,
                        "move_group_result_timeout": move_group_result_timeout,
                        "direct_trajectory_result_timeout": direct_trajectory_result_timeout,
                        "direct_trajectory_goal_time_tolerance": direct_trajectory_goal_time_tolerance,
                        "arm_state_topic": arm_state_topic,
                        "coordinator_state_topic": coordinator_state_topic,
                        "reset_topic": reset_topic,
                        "robot_xrdf": collision_robot_xrdf,
                        "robot_urdf": collision_robot_urdf_path,
                    }
                ],
            ),
            Node(
                condition=IfCondition(launch_left_planner),
                package="motion",
                executable="planner",
                name="motion_planner_left",
                output="screen",
                parameters=[
                    {
                        "planning_group": planning_group,
                        "planning_arm_side": "left",
                        "pipeline_id": pipeline_id,
                        "planner_id": planner_id,
                        "move_group_action": "/moveit_left/move_action",
                        "planning_scene_service": "/moveit_left/get_planning_scene",
                        "planning_scene_topic": "/moveit_left/planning_scene",
                        "plan_only": plan_only,
                        "move_group_result_timeout": move_group_result_timeout,
                        "direct_trajectory_result_timeout": direct_trajectory_result_timeout,
                        "direct_trajectory_goal_time_tolerance": direct_trajectory_goal_time_tolerance,
                        "arm_state_topic": arm_state_topic,
                        "coordinator_state_topic": coordinator_state_topic,
                        "reset_topic": reset_topic,
                        "robot_xrdf": collision_robot_xrdf,
                        "robot_urdf": collision_robot_urdf_path,
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
                        "robot_xrdf": right_cumotion_robot_xrdf,
                        "urdf_path": right_cumotion_urdf_path,
                        "joint_states_topic": "/joint_states",
                        "marker_topic": "/cumotion/robot_spheres",
                        "frame_id": "yumi_body",
                    }
                ],
            ),
        ]
    )
