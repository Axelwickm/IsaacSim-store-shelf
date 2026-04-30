from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from moveit_configs_utils import MoveItConfigsBuilder


ISAAC_JOINT_STATES_TOPIC = "/isaac_joint_states"
JOINT_COMMAND_TOPIC = "/joint_command"
CONTROLLER_MANAGER = "/controller_manager"
SPAWNER_CONTROLLER_MANAGER_TIMEOUT = "30"
SPAWNER_SWITCH_TIMEOUT = "60"


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


def _namespaced_service(namespace: str, service_name: str) -> str:
    name = service_name.strip("/")
    namespace = namespace.strip("/")
    if not namespace:
        return f"/{name}"
    return f"/{namespace}/{name}"


def _cumotion_launch_path() -> str:
    return str(
        Path(get_package_share_directory("isaac_ros_cumotion"))
        / "launch"
        / "isaac_ros_cumotion.launch.py"
    )


def _bool_arg(context, name: str) -> bool:
    value = LaunchConfiguration(name).perform(context).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _side_robot_description(package_share: Path, side: str) -> dict[str, Command]:
    yumi_description_xacro = package_share / "urdf" / "yumi.urdf.xacro"
    include_left = "true" if side == "left" else "false"
    include_right = "true" if side == "right" else "false"
    return {
        "robot_description": Command(
            [
                "xacro",
                " ",
                str(yumi_description_xacro),
                " ",
                f"include_left_arm:={include_left}",
                " ",
                f"include_right_arm:={include_right}",
                " ",
                f"include_left_gripper:={include_left}",
                " ",
                f"include_right_gripper:={include_right}",
                " ",
                "use_ros2_control:=false",
            ]
        )
    }


def _runtime_robot_description(package_share: Path) -> dict[str, Command]:
    yumi_description_xacro = package_share / "urdf" / "yumi.urdf.xacro"
    return {
        "robot_description": Command(
            [
                "xacro",
                " ",
                str(yumi_description_xacro),
                " ",
                "use_ros2_control:=true",
                " ",
                f"joint_states_topic:={ISAAC_JOINT_STATES_TOPIC}",
                " ",
                f"joint_commands_topic:={JOINT_COMMAND_TOPIC}",
            ]
        )
    }


def _moveit_config_for_side(side: str):
    semantic_file = (
        "config/yumi_left_arm.srdf" if side == "left" else "config/yumi_right_arm.srdf"
    )
    return (
        MoveItConfigsBuilder("yumi", package_name="yumi_moveit_config")
        .robot_description_semantic(file_path=semantic_file)
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


def _build_launch_entities(context):
    planning_pipeline = LaunchConfiguration("planning_pipeline").perform(context)
    use_rviz = _bool_arg(context, "use_rviz")
    use_sim_time = _bool_arg(context, "use_sim_time")
    launch_runtime_support = _bool_arg(context, "launch_runtime_support")
    controller_spawner_delay = float(
        LaunchConfiguration("controller_spawner_delay").perform(context)
    )
    move_group_delay = float(LaunchConfiguration("move_group_delay").perform(context))
    move_group_log_level = LaunchConfiguration("move_group_log_level").perform(context)
    cumotion_robot_xrdf = LaunchConfiguration("cumotion_robot_xrdf").perform(context)
    cumotion_urdf_path = LaunchConfiguration("cumotion_urdf_path").perform(context)
    planning_arm_side = (
        LaunchConfiguration("planning_arm_side").perform(context).strip().lower()
    )
    move_group_namespace = (
        LaunchConfiguration("move_group_namespace").perform(context).strip().strip("/")
    )

    if planning_arm_side not in {"left", "right"}:
        raise ValueError("planning_arm_side must be 'left' or 'right'")

    yumi_description_share = Path(get_package_share_directory("yumi_description"))
    yumi_moveit_config_share = Path(get_package_share_directory("yumi_moveit_config"))

    planning_robot_description = _side_robot_description(
        yumi_description_share, planning_arm_side
    )
    runtime_robot_description = _runtime_robot_description(yumi_description_share)
    moveit_config = _moveit_config_for_side(planning_arm_side)
    ros2_controllers_path = str(
        yumi_moveit_config_share / "config" / "ros2_controllers.yaml"
    )
    rviz_config_path = str(
        yumi_moveit_config_share / "rviz" / "cumotion_debug.rviz"
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
    get_planning_scene_service = _namespaced_service(
        move_group_namespace, "get_planning_scene"
    )
    apply_planning_scene_service = _namespaced_service(
        move_group_namespace, "apply_planning_scene"
    )
    namespace_value = move_group_namespace or None

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=namespace_value,
        output="screen",
        arguments=["--ros-args", "--log-level", move_group_log_level],
        parameters=[
            moveit_config.to_dict(),
            planning_robot_description,
            {"default_planning_pipeline": planning_pipeline},
            planning_scene_monitor_parameters,
            start_state_bounds_parameters,
            {"use_sim_time": use_sim_time},
        ],
    )
    static_planning_scene_node = Node(
        package="yumi_moveit_config",
        executable="static_planning_scene",
        namespace=namespace_value,
        name="static_planning_scene",
        output="screen",
        parameters=[
            {
                "collision_config": str(
                    yumi_moveit_config_share / "config" / "store_shelf_collision.yaml"
                ),
                "apply_planning_scene_service": apply_planning_scene_service,
                "cumotion_scene_service": (
                    "/publish_static_planning_scene" if launch_runtime_support else ""
                ),
                "planning_scene_topics": [
                    "/planning_scene",
                    "/moveit_left/planning_scene",
                    "/moveit_right/planning_scene",
                ],
            }
        ],
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            runtime_robot_description,
            ros2_controllers_path,
            {"use_sim_time": False},
        ],
    )
    joint_state_broadcaster_spawner = controller_spawner("joint_state_broadcaster")
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[runtime_robot_description, {"use_sim_time": use_sim_time}],
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
        parameters=[
            planning_robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
        ],
    )

    cumotion_enabled = (
        planning_pipeline == "isaac_ros_cumotion"
        and cumotion_robot_xrdf
        and cumotion_urdf_path
    )
    cumotion_ready = PythonExpression(
        [
            "'",
            planning_pipeline,
            "' == 'isaac_ros_cumotion' and '",
            cumotion_robot_xrdf,
            "' != '' and '",
            cumotion_urdf_path,
            "' != ''",
        ]
    )
    cumotion_namespace_actions = []
    if namespace_value:
        cumotion_namespace_actions.append(PushRosNamespace(namespace_value))
    cumotion_namespace_actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(_cumotion_launch_path()),
            launch_arguments={
                "cumotion_planner.robot": cumotion_robot_xrdf,
                "cumotion_planner.urdf_path": cumotion_urdf_path,
            }.items(),
        )
    )
    cumotion_planner_launch = GroupAction(
        actions=cumotion_namespace_actions,
        condition=IfCondition(cumotion_ready),
    )

    launch_entities = [
        LogInfo(
            msg=(
                f"Launching {planning_arm_side}-arm MoveIt stack "
                f"(namespace=/{move_group_namespace or ''}, "
                f"planning_scene_service={get_planning_scene_service})."
            )
        )
    ]

    if launch_runtime_support:
        launch_entities.extend(
            [
                robot_state_publisher_node,
                ros2_control_node,
                TimerAction(
                    period=controller_spawner_delay,
                    actions=[joint_state_broadcaster_spawner],
                ),
            ]
        )
    if cumotion_enabled:
        launch_entities.append(
            LogInfo(
                msg=(
                    f"Will launch {planning_arm_side}-arm cuMotion stack in "
                    f"/{move_group_namespace or ''} with XRDF={cumotion_robot_xrdf} "
                    f"URDF={cumotion_urdf_path} after move_group startup."
                )
            )
        )
    elif planning_pipeline == "isaac_ros_cumotion":
        launch_entities.append(
            LogInfo(
                msg=(
                    f"Skipping {planning_arm_side}-arm cuMotion stack because "
                    "cumotion_robot_xrdf or cumotion_urdf_path is empty."
                )
            )
        )

    start_stack_actions = [move_group_node]
    if cumotion_enabled:
        start_stack_actions.append(
            TimerAction(period=2.0, actions=[cumotion_planner_launch])
        )
    start_stack_actions.append(
        TimerAction(period=2.0, actions=[static_planning_scene_node])
    )
    if launch_runtime_support:
        launch_entities.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=joint_state_broadcaster_spawner,
                    on_exit=[
                        TimerAction(
                            period=move_group_delay,
                            actions=start_stack_actions,
                        )
                    ],
                )
            )
        )
    else:
        launch_entities.append(
            TimerAction(
                period=controller_spawner_delay + move_group_delay,
                actions=start_stack_actions,
            )
        )

    if use_rviz:
        launch_entities.append(rviz_node)
    return launch_entities


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "planning_pipeline",
                default_value="ompl",
                description="MoveIt planning pipeline to use: ompl or isaac_ros_cumotion.",
            ),
            DeclareLaunchArgument(
                "planning_arm_side",
                default_value="right",
                description="Planning arm side: left or right.",
            ),
            DeclareLaunchArgument(
                "move_group_namespace",
                default_value="moveit_right",
                description="ROS namespace for the move_group and cuMotion stack.",
            ),
            DeclareLaunchArgument(
                "launch_runtime_support",
                default_value="true",
                description="Launch shared ros2_control and robot_state_publisher support.",
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
                description="Seconds to wait before starting move_group.",
            ),
            DeclareLaunchArgument(
                "move_group_log_level",
                default_value="info",
                description="ROS log level for move_group.",
            ),
            DeclareLaunchArgument(
                "cumotion_robot_xrdf",
                default_value="/workspace/usd/robot/yumi_isaacsim_right_arm.xrdf",
                description="XRDF file passed to the standalone cuMotion planner node.",
            ),
            DeclareLaunchArgument(
                "cumotion_urdf_path",
                default_value="/workspace/usd/robot/yumi_isaacsim_right_arm.urdf",
                description="URDF file passed to the standalone cuMotion planner node.",
            ),
            OpaqueFunction(function=_build_launch_entities),
        ]
    )
