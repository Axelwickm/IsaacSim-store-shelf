from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


ISAAC_JOINT_STATES_TOPIC = "/isaac_joint_states"
JOINT_COMMAND_TOPIC = "/joint_command"
CONTROLLER_MANAGER = "/controller_manager"
SPAWNER_CONTROLLER_MANAGER_TIMEOUT = "30"
SPAWNER_SWITCH_TIMEOUT = "60"


def load_yaml(package_name: str, relative_path: str):
    import yaml

    package_share = Path(get_package_share_directory(package_name))
    with (package_share / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def controller_spawner(controller_name: str) -> Node:
    return Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            controller_name,
            "--controller-manager",
            CONTROLLER_MANAGER,
            "--controller-manager-timeout",
            SPAWNER_CONTROLLER_MANAGER_TIMEOUT,
            "--switch-timeout",
            SPAWNER_SWITCH_TIMEOUT,
        ],
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    planning_pipeline = LaunchConfiguration("planning_pipeline")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    controller_spawner_delay = LaunchConfiguration("controller_spawner_delay")
    move_group_delay = LaunchConfiguration("move_group_delay")

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
                "use_ros2_control": "true",
                "joint_states_topic": ISAAC_JOINT_STATES_TOPIC,
                "joint_commands_topic": JOINT_COMMAND_TOPIC,
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
    ros2_controllers_path = str(
        Path(get_package_share_directory("yumi_moveit_config"))
        / "config"
        / "ros2_controllers.yaml"
    )
    planning_scene_monitor_parameters = {
        "publish_planning_scene": False,
        "publish_geometry_updates": False,
        "publish_state_updates": False,
        "publish_transforms_updates": False,
        "monitor_dynamics": False,
    }
    start_state_bounds_parameters = {
        "start_state_max_bounds_error": 0.02,
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
            start_state_bounds_parameters,
            {"use_sim_time": use_sim_time},
        ],
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            ros2_controllers_path,
            {"use_sim_time": False},
        ],
    )

    joint_state_broadcaster_spawner = controller_spawner("joint_state_broadcaster")
    right_arm_controller_spawner = controller_spawner("right_arm_controller")

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
                "move_group controller discovery."
            ),
        ),
        robot_state_publisher_node,
        ros2_control_node,
        TimerAction(
            period=controller_spawner_delay,
            actions=[joint_state_broadcaster_spawner],
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[right_arm_controller_spawner],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=right_arm_controller_spawner,
                on_exit=[
                    TimerAction(period=move_group_delay, actions=[move_group_node])
                ],
            )
        ),
        rviz_node,
    ]

    return LaunchDescription(launch_entities)
