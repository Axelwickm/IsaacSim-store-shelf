from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _moveit_launch_path() -> str:
    return str(
        Path(get_package_share_directory("yumi_moveit_config"))
        / "launch"
        / "move_group.launch.py"
    )


def generate_launch_description() -> LaunchDescription:
    headless = LaunchConfiguration("headless")
    configuration = LaunchConfiguration("configuration")
    use_moveit = LaunchConfiguration("use_moveit")
    planning_pipeline = LaunchConfiguration("planning_pipeline")
    use_moveit_rviz = LaunchConfiguration("use_moveit_rviz")
    controller_spawner_delay = LaunchConfiguration("controller_spawner_delay")
    move_group_delay = LaunchConfiguration("move_group_delay")
    move_group_log_level = LaunchConfiguration("move_group_log_level")
    cumotion_robot_xrdf = LaunchConfiguration("cumotion_robot_xrdf")
    cumotion_urdf_path = LaunchConfiguration("cumotion_urdf_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Whether Isaac Sim should launch in headless mode.",
            ),
            DeclareLaunchArgument(
                "configuration",
                default_value="",
                description="Named simulation configuration to load.",
            ),
            DeclareLaunchArgument(
                "use_moveit",
                default_value="true",
                description="Whether to launch the MoveIt subsystem alongside Isaac Sim.",
            ),
            DeclareLaunchArgument(
                "planning_pipeline",
                default_value="ompl",
                description="MoveIt planning pipeline to use: ompl or isaac_ros_cumotion.",
            ),
            DeclareLaunchArgument(
                "use_moveit_rviz",
                default_value="false",
                description="Whether to launch RViz for the MoveIt subsystem.",
            ),
            DeclareLaunchArgument(
                "controller_spawner_delay",
                default_value="30.0",
                description="Seconds to wait before activating ros2_control controllers.",
            ),
            DeclareLaunchArgument(
                "move_group_delay",
                default_value="5.0",
                description=(
                    "Seconds to wait after controller spawners finish before starting "
                    "move_group."
                ),
            ),
            DeclareLaunchArgument(
                "move_group_log_level",
                default_value="info",
                description="ROS log level for move_group.",
            ),
            DeclareLaunchArgument(
                "cumotion_robot_xrdf",
                default_value="/workspace/usd/robot/yumi_isaacsim.xrdf",
                description="XRDF file for the standalone cuMotion planner node.",
            ),
            DeclareLaunchArgument(
                "cumotion_urdf_path",
                default_value="/workspace/usd/robot/yumi_isaacsim.urdf",
                description="URDF file for the standalone cuMotion planner node.",
            ),
            Node(
                package="controller",
                executable="controller",
                name="controller",
                output="screen",
                parameters=[
                    {
                        "headless": headless,
                        "configuration": configuration,
                    }
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(_moveit_launch_path()),
                condition=IfCondition(use_moveit),
                launch_arguments={
                    "planning_pipeline": planning_pipeline,
                    "use_rviz": use_moveit_rviz,
                    "use_sim_time": "true",
                    "controller_spawner_delay": controller_spawner_delay,
                    "move_group_delay": move_group_delay,
                    "move_group_log_level": move_group_log_level,
                    "cumotion_robot_xrdf": cumotion_robot_xrdf,
                    "cumotion_urdf_path": cumotion_urdf_path,
                }.items(),
            ),
        ]
    )
