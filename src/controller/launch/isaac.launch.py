from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    headless = LaunchConfiguration("headless")
    configuration = LaunchConfiguration("configuration")

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
        ]
    )
