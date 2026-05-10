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
                description="Optional explicit checkpoint path to load before training.",
            ),
            DeclareLaunchArgument(
                "dataset_dir",
                default_value="/workspace/collect_vision_data_output",
                description="Collected Replicator dataset directory.",
            ),
            DeclareLaunchArgument(
                "replay_dir",
                default_value="/workspace/replay/vision",
                description="Planner feedback replay directory.",
            ),
            DeclareLaunchArgument(
                "image_size",
                default_value="512",
                description="Square image size used by the vision model.",
            ),
            DeclareLaunchArgument(
                "replay_buffer_capacity",
                default_value="512",
                description="Maximum samples kept in each in-memory training buffer.",
            ),
            DeclareLaunchArgument(
                "min_replay_size",
                default_value="2",
                description="Minimum collected samples required before training starts.",
            ),
            DeclareLaunchArgument(
                "train_batch_size",
                default_value="2",
                description="Training batch size.",
            ),
            DeclareLaunchArgument(
                "train_steps_per_tick",
                default_value="1",
                description="Optimizer steps per timer tick.",
            ),
            DeclareLaunchArgument(
                "train_tick_period_sec",
                default_value="0.05",
                description="Seconds between optimizer timer ticks.",
            ),
            DeclareLaunchArgument(
                "online_learning_rate",
                default_value="2e-4",
                description="Learning rate for standalone online training.",
            ),
            DeclareLaunchArgument(
                "geometry_loss_weight",
                default_value="1.0",
                description="Weight for billboard identity and geometry supervision.",
            ),
            DeclareLaunchArgument(
                "presence_loss_weight",
                default_value="0.2",
                description="Weight for missed-occupancy supervision.",
            ),
            DeclareLaunchArgument(
                "depth_loss_weight",
                default_value="0.2",
                description="Weight for billboard depth supervision.",
            ),
            DeclareLaunchArgument(
                "value_loss_weight",
                default_value="1.0",
                description="Weight for arm-conditioned planner-success supervision.",
            ),
            DeclareLaunchArgument(
                "use_mixed_precision",
                default_value="true",
                description="Whether to use CUDA mixed precision.",
            ),
            DeclareLaunchArgument(
                "tensorboard_log_dir",
                default_value="/workspace/tensorboard/vision",
                description="TensorBoard base log directory.",
            ),
            DeclareLaunchArgument(
                "tensorboard_run_name",
                default_value="",
                description="Optional TensorBoard run subdirectory name.",
            ),
            DeclareLaunchArgument(
                "checkpoint_save_interval",
                default_value="100",
                description="Optimizer steps between checkpoint writes.",
            ),
            DeclareLaunchArgument(
                "replay_scan_period_sec",
                default_value="1.0",
                description="Seconds between dataset/replay directory scans.",
            ),
            DeclareLaunchArgument(
                "sample_prefetch_size",
                default_value="64",
                description="Number of decoded Replicator samples to keep ready for training.",
            ),
            DeclareLaunchArgument(
                "sample_loader_workers",
                default_value="2",
                description="Background workers used to decode Replicator samples.",
            ),
            DeclareLaunchArgument(
                "training_debug_period_steps",
                default_value="25",
                description="Optimizer steps between local training visualization updates.",
            ),
            DeclareLaunchArgument(
                "cuda_memory_log_period_sec",
                default_value="30.0",
                description="Seconds between trainer CUDA memory logs.",
            ),
            Node(
                package="vision",
                executable="vision_online_trainer",
                name="vision_online_trainer",
                output="screen",
                parameters=[
                    {
                        "checkpoint_dir": LaunchConfiguration("checkpoint_dir"),
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "dataset_dir": LaunchConfiguration("dataset_dir"),
                        "replay_dir": LaunchConfiguration("replay_dir"),
                        "image_size": LaunchConfiguration("image_size"),
                        "replay_buffer_capacity": LaunchConfiguration("replay_buffer_capacity"),
                        "min_replay_size": LaunchConfiguration("min_replay_size"),
                        "train_batch_size": LaunchConfiguration("train_batch_size"),
                        "train_steps_per_tick": LaunchConfiguration("train_steps_per_tick"),
                        "train_tick_period_sec": LaunchConfiguration("train_tick_period_sec"),
                        "online_learning_rate": LaunchConfiguration("online_learning_rate"),
                        "geometry_loss_weight": LaunchConfiguration("geometry_loss_weight"),
                        "presence_loss_weight": LaunchConfiguration("presence_loss_weight"),
                        "depth_loss_weight": LaunchConfiguration("depth_loss_weight"),
                        "value_loss_weight": LaunchConfiguration("value_loss_weight"),
                        "use_mixed_precision": LaunchConfiguration("use_mixed_precision"),
                        "tensorboard_log_dir": LaunchConfiguration("tensorboard_log_dir"),
                        "tensorboard_run_name": LaunchConfiguration("tensorboard_run_name"),
                        "checkpoint_save_interval": LaunchConfiguration("checkpoint_save_interval"),
                        "replay_scan_period_sec": LaunchConfiguration("replay_scan_period_sec"),
                        "sample_prefetch_size": LaunchConfiguration("sample_prefetch_size"),
                        "sample_loader_workers": LaunchConfiguration("sample_loader_workers"),
                        "training_debug_period_steps": LaunchConfiguration(
                            "training_debug_period_steps"
                        ),
                        "cuda_memory_log_period_sec": LaunchConfiguration(
                            "cuda_memory_log_period_sec"
                        ),
                    }
                ],
            ),
        ]
    )
