from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "checkpoint_dir",
                default_value="/workspace/checkpoints/vision",
                description="Directory where latest and epoch checkpoints are stored.",
            ),
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value="",
                description="Optional explicit checkpoint path to load.",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/camera/image_raw",
                description="Camera topic providing live RGB images.",
            ),
            DeclareLaunchArgument(
                "debug_image_topic",
                default_value="/vision/debug_image",
                description="Composite debug output topic for Isaac Sim or other viewers.",
            ),
            DeclareLaunchArgument(
                "identity_topic",
                default_value="/vision/predicted_identity",
                description="Debug output topic for predicted identity visualization.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/vision/predicted_depth",
                description="Output topic for predicted depth images in 32FC1 format.",
            ),
            DeclareLaunchArgument(
                "depth_viz_topic",
                default_value="/vision/predicted_depth_viz",
                description="Debug output topic for colorized predicted depth.",
            ),
            DeclareLaunchArgument(
                "occupancy_topic",
                default_value="/vision/predicted_occupancy",
                description="Debug output topic for predicted occupancy visualization.",
            ),
            DeclareLaunchArgument(
                "image_size",
                default_value="512",
                description="Square image size used by the vision model.",
            ),
            DeclareLaunchArgument(
                "use_mixed_precision",
                default_value="true",
                description="Whether to enable CUDA mixed precision.",
            ),
            Node(
                package="vision",
                executable="vision_inference",
                name="vision_inference",
                output="screen",
                parameters=[
                    {
                        "checkpoint_dir": LaunchConfiguration("checkpoint_dir"),
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "debug_image_topic": LaunchConfiguration("debug_image_topic"),
                        "identity_topic": LaunchConfiguration("identity_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "depth_viz_topic": LaunchConfiguration("depth_viz_topic"),
                        "occupancy_topic": LaunchConfiguration("occupancy_topic"),
                        "image_size": LaunchConfiguration("image_size"),
                        "use_mixed_precision": LaunchConfiguration("use_mixed_precision"),
                    }
                ],
            ),
        ]
    )
