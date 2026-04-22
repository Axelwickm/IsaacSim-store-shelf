from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    headless = LaunchConfiguration("headless")
    isaac_launch = PathJoinSubstitution(
        [FindPackageShare("controller"), "launch", "isaac.launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="true",
                description="Whether Isaac Sim should launch in headless mode.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(isaac_launch),
                launch_arguments={
                    "headless": headless,
                    "configuration": "collect_vision_data",
                }.items(),
            ),
        ]
    )
