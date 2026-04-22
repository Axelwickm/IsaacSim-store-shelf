from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    headless = LaunchConfiguration("headless")
    scene = LaunchConfiguration("scene")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Whether Isaac Sim should launch in headless mode.",
            ),
            DeclareLaunchArgument(
                "scene",
                default_value="",
                description="Scene or USD file to load when Isaac Sim starts.",
            ),
            Node(
                package="controller",
                executable="controller",
                name="controller",
                output="screen",
                parameters=[
                    {
                        "headless": headless,
                        "scene": scene,
                    }
                ],
            ),
        ]
    )
