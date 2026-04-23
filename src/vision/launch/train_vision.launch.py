from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "dataset_dir",
                default_value="/workspace/collect_vision_data_output",
                description="Directory containing collected vision samples.",
            ),
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
                "batch_size",
                default_value="1",
                description="Batch size for vision model training or evaluation.",
            ),
            DeclareLaunchArgument(
                "split",
                default_value="train",
                description="Dataset split to use: train or test.",
            ),
            DeclareLaunchArgument(
                "mode",
                default_value="train",
                description="Whether to run training or evaluation.",
            ),
            DeclareLaunchArgument(
                "learning_rate",
                default_value="0.00001",
                description="AdamW learning rate used in train mode.",
            ),
            DeclareLaunchArgument(
                "resume",
                default_value="true",
                description="Whether train mode should resume from the latest checkpoint.",
            ),
            DeclareLaunchArgument(
                "save_every_epochs",
                default_value="1",
                description="How often to write numbered epoch checkpoints.",
            ),
            DeclareLaunchArgument(
                "image_size",
                default_value="512",
                description="Square image size used for model inputs.",
            ),
            DeclareLaunchArgument(
                "use_mixed_precision",
                default_value="true",
                description="Whether to enable CUDA mixed precision.",
            ),
            Node(
                package="vision",
                executable="train_vision",
                name="vision_trainer",
                output="screen",
                parameters=[
                    {
                        "dataset_dir": LaunchConfiguration("dataset_dir"),
                        "checkpoint_dir": LaunchConfiguration("checkpoint_dir"),
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "batch_size": LaunchConfiguration("batch_size"),
                        "split": LaunchConfiguration("split"),
                        "mode": LaunchConfiguration("mode"),
                        "learning_rate": LaunchConfiguration("learning_rate"),
                        "resume": LaunchConfiguration("resume"),
                        "save_every_epochs": LaunchConfiguration("save_every_epochs"),
                        "image_size": LaunchConfiguration("image_size"),
                        "use_mixed_precision": LaunchConfiguration("use_mixed_precision"),
                    }
                ],
            ),
        ]
    )
