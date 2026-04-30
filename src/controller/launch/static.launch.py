from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    headless = LaunchConfiguration("headless")
    use_rviz = LaunchConfiguration("use_rviz")
    publish_cumotion_spheres = LaunchConfiguration("publish_cumotion_spheres")
    cumotion_robot_xrdf = LaunchConfiguration("cumotion_robot_xrdf")
    cumotion_urdf_path = LaunchConfiguration("cumotion_urdf_path")
    sim_launch = PathJoinSubstitution(
        [FindPackageShare("controller"), "launch", "sim.launch.py"]
    )
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("yumi_moveit_config"), "rviz", "cumotion_debug.rviz"]
    )
    yumi_moveit_config_share = Path(get_package_share_directory("yumi_moveit_config"))
    yumi_description_share = Path(get_package_share_directory("yumi_description"))
    robot_description_content = ParameterValue(
        Command(["xacro ", str(yumi_description_share / "urdf" / "yumi.urdf.xacro")]),
        value_type=str,
    )
    robot_description_semantic = {
        "robot_description_semantic": (
            yumi_moveit_config_share / "config" / "yumi.srdf"
        ).read_text(encoding="utf-8")
    }
    robot_description_kinematics = {
        "robot_description_kinematics": {
            "left_arm": {
                "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
                "kinematics_solver_search_resolution": 0.005,
                "kinematics_solver_timeout": 0.005,
            },
            "right_arm": {
                "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
                "kinematics_solver_search_resolution": 0.005,
                "kinematics_solver_timeout": 0.005,
            },
        }
    }
    planning_pipelines = {
        "planning_pipelines": ["ompl", "isaac_ros_cumotion"],
        "default_planning_pipeline": "ompl",
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="true",
                description="Whether Isaac Sim should launch in headless mode.",
            ),
            DeclareLaunchArgument(
                "publish_cumotion_spheres",
                default_value="true",
                description="Publish cuMotion robot collision spheres for RViz debugging.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Launch RViz with the cuMotion sphere debug config.",
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
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sim_launch),
                launch_arguments={
                    "headless": headless,
                    "configuration": "static",
                    "use_moveit": "false",
                }.items(),
            ),
            Node(
                condition=IfCondition(publish_cumotion_spheres),
                package="motion",
                executable="cumotion_sphere_publisher",
                name="cumotion_sphere_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_xrdf": cumotion_robot_xrdf,
                        "urdf_path": cumotion_urdf_path,
                        "joint_states_topic": "/isaac_joint_states",
                        "marker_topic": "/cumotion/robot_spheres",
                        "frame_id": "yumi_body",
                    }
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[
                    {"robot_description": robot_description_content},
                    {"use_sim_time": True},
                ],
                remappings=[("/joint_states", "/isaac_joint_states")],
            ),
            Node(
                condition=IfCondition(use_rviz),
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                parameters=[
                    {"robot_description": robot_description_content},
                    robot_description_semantic,
                    robot_description_kinematics,
                    planning_pipelines,
                ],
            ),
        ]
    )
