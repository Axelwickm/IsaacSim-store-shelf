from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml(package_name: str, relative_path: str):
    import yaml

    package_share = Path(get_package_share_directory(package_name))
    with (package_share / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_launch_description() -> LaunchDescription:
    planning_pipeline = LaunchConfiguration("planning_pipeline")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")

    moveit_config = (
        MoveItConfigsBuilder("yumi", package_name="yumi_moveit_config")
        .robot_description(
            file_path=str(
                Path(get_package_share_directory("yumi_description"))
                / "urdf"
                / "yumi.urdf.xacro"
            ),
            mappings={
                "arms_interface": "PositionJointInterface",
                "grippers_interface": "PositionJointInterface",
            },
        )
        .robot_description_semantic(file_path="config/yumi.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl", "isaac_ros_cumotion"],
        )
        .to_moveit_configs()
    )
    trajectory_execution = load_yaml(
        "yumi_moveit_config", "config/trajectory_execution.yaml"
    )
    moveit_controllers = load_yaml(
        "yumi_moveit_config", "config/moveit_controllers.yaml"
    )
    planning_scene_monitor_parameters = {
        "publish_planning_scene": False,
        "publish_geometry_updates": False,
        "publish_state_updates": False,
        "publish_transforms_updates": False,
        "monitor_dynamics": False,
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"default_planning_pipeline": planning_pipeline},
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            {"use_sim_time": use_sim_time},
        ],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description, {"use_sim_time": use_sim_time}],
    )

    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package="rviz2",
        executable="rviz2",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
        ],
    )

    launch_entities = [
        DeclareLaunchArgument(
            "planning_pipeline",
            default_value="ompl",
            description="MoveIt planning pipeline to use: ompl or isaac_ros_cumotion.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Whether to launch RViz alongside move_group.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use Isaac Sim /clock for MoveIt state freshness checks.",
        ),
        robot_state_publisher_node,
        move_group_node,
        rviz_node,
    ]

    return LaunchDescription(launch_entities)
