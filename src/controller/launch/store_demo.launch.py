from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    sim_launch = Path(get_package_share_directory("controller")) / "launch" / "sim.launch.py"
    vision_launch = Path(get_package_share_directory("vision")) / "launch" / "inference.launch.py"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "vision_checkpoint_dir",
                default_value="/workspace/checkpoints/vision",
                description="Directory containing the latest vision checkpoint.",
            ),
            DeclareLaunchArgument(
                "vision_checkpoint_path",
                default_value="",
                description="Optional explicit vision checkpoint path for inference.",
            ),
            DeclareLaunchArgument(
                "vision_image_size",
                default_value="512",
                description="Square image size used by the vision model.",
            ),
            DeclareLaunchArgument(
                "vision_use_mixed_precision",
                default_value="true",
                description="Whether to enable CUDA mixed precision for vision inference.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(sim_launch)),
                launch_arguments={
                    "configuration": "store_demo",
                    "use_moveit": "true",
                    "planning_pipeline": "ompl",
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(vision_launch)),
                launch_arguments={
                    "checkpoint_dir": LaunchConfiguration("vision_checkpoint_dir"),
                    "checkpoint_path": LaunchConfiguration("vision_checkpoint_path"),
                    "image_topic": "/camera/image_raw",
                    "image_size": LaunchConfiguration("vision_image_size"),
                    "use_mixed_precision": LaunchConfiguration(
                        "vision_use_mixed_precision"
                    ),
                }.items(),
            ),
        ]
    )
