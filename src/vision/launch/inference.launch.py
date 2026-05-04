from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
                description="RGB debug image with predicted candidate dots and labels.",
            ),
            DeclareLaunchArgument(
                "image_size",
                default_value="512",
                description="Square image size used by the vision model.",
            ),
            DeclareLaunchArgument(
                "selected_candidate_topic",
                default_value="/vision/selected_candidate",
                description="Atomic candidate topic containing arm choice and 3D target metadata.",
            ),
            DeclareLaunchArgument(
                "suggested_item_markers_topic",
                default_value="/vision/suggested_item_markers",
                description="RViz MarkerArray topic for multiple projected suggested 3D item points.",
            ),
            DeclareLaunchArgument(
                "ground_truth_items_topic",
                default_value="/vision/ground_truth_items",
                description="IsaacSim-published item world centers used for green debug markers.",
            ),
            DeclareLaunchArgument(
                "arm_state_topic",
                default_value="/motion/arm_state",
                description="Planner arm-state topic used for online replay labels.",
            ),
            DeclareLaunchArgument(
                "online_training_enabled",
                default_value="false",
                description="Record replay samples and launch the external online trainer.",
            ),
            DeclareLaunchArgument(
                "replay_dir",
                default_value="/workspace/replay/vision",
                description="Directory used as the durable replay-sample queue.",
            ),
            DeclareLaunchArgument(
                "checkpoint_reload_period_sec",
                default_value="5.0",
                description="Seconds between inference checkpoint hot-reload checks.",
            ),
            DeclareLaunchArgument(
                "exploration_epsilon",
                default_value="0.10",
                description="Probability of exploring a random valid query/arm pair.",
            ),
            DeclareLaunchArgument(
                "query_presence_threshold",
                default_value="0.20",
                description="Minimum predicted presence probability for a candidate to be selectable.",
            ),
            DeclareLaunchArgument(
                "max_suggested_markers",
                default_value="32",
                description="Maximum projected suggested 3D item markers to publish.",
            ),
            DeclareLaunchArgument(
                "camera_frame_convention",
                default_value="ros_optical",
                description="Camera TF convention: usd_camera or ros_optical.",
            ),
            DeclareLaunchArgument(
                "replay_buffer_capacity",
                default_value="2048",
                description="Maximum number of replay samples kept in memory.",
            ),
            DeclareLaunchArgument(
                "min_replay_size",
                default_value="2",
                description="Minimum replay samples required before online training begins.",
            ),
            DeclareLaunchArgument(
                "train_batch_size",
                default_value="16",
                description="Batch size for online replay-buffer updates.",
            ),
            DeclareLaunchArgument(
                "train_steps_per_tick",
                default_value="4",
                description="Maximum online optimizer steps to run per timer tick.",
            ),
            DeclareLaunchArgument(
                "train_tick_period_sec",
                default_value="0.01",
                description="Seconds between online optimizer timer ticks.",
            ),
            DeclareLaunchArgument(
                "online_learning_rate",
                default_value="1e-3",
                description="Learning rate for online replay-buffer training.",
            ),
            DeclareLaunchArgument(
                "geometry_loss_weight",
                default_value="1.0",
                description="Weight for online GT-projected query-center supervision.",
            ),
            DeclareLaunchArgument(
                "presence_loss_weight",
                default_value="0.2",
                description="Weight for online GT query-presence supervision.",
            ),
            DeclareLaunchArgument(
                "depth_loss_weight",
                default_value="0.2",
                description="Weight for online GT projected-depth supervision.",
            ),
            DeclareLaunchArgument(
                "tensorboard_log_dir",
                default_value="/workspace/tensorboard/vision",
                description="TensorBoard base log directory used only when online training is enabled.",
            ),
            DeclareLaunchArgument(
                "tensorboard_run_name",
                default_value="",
                description=(
                    "Optional TensorBoard run subdirectory name. If empty, a unique "
                    "timestamped name is generated or reused from a loaded checkpoint."
                ),
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
                        "selected_candidate_topic": LaunchConfiguration("selected_candidate_topic"),
                        "suggested_item_markers_topic": LaunchConfiguration(
                            "suggested_item_markers_topic"
                        ),
                        "ground_truth_items_topic": LaunchConfiguration(
                            "ground_truth_items_topic"
                        ),
                        "arm_state_topic": LaunchConfiguration("arm_state_topic"),
                        "online_training_enabled": LaunchConfiguration("online_training_enabled"),
                        "replay_dir": LaunchConfiguration("replay_dir"),
                        "checkpoint_reload_period_sec": LaunchConfiguration(
                            "checkpoint_reload_period_sec"
                        ),
                        "exploration_epsilon": LaunchConfiguration("exploration_epsilon"),
                        "query_presence_threshold": LaunchConfiguration("query_presence_threshold"),
                        "max_suggested_markers": LaunchConfiguration("max_suggested_markers"),
                        "camera_frame_convention": LaunchConfiguration("camera_frame_convention"),
                        "replay_buffer_capacity": LaunchConfiguration("replay_buffer_capacity"),
                        "min_replay_size": LaunchConfiguration("min_replay_size"),
                        "train_batch_size": LaunchConfiguration("train_batch_size"),
                        "online_learning_rate": LaunchConfiguration("online_learning_rate"),
                        "geometry_loss_weight": LaunchConfiguration("geometry_loss_weight"),
                        "presence_loss_weight": LaunchConfiguration("presence_loss_weight"),
                        "depth_loss_weight": LaunchConfiguration("depth_loss_weight"),
                        "tensorboard_log_dir": LaunchConfiguration("tensorboard_log_dir"),
                        "tensorboard_run_name": LaunchConfiguration("tensorboard_run_name"),
                        "image_size": LaunchConfiguration("image_size"),
                        "use_mixed_precision": LaunchConfiguration("use_mixed_precision"),
                    }
                ],
            ),
            Node(
                condition=IfCondition(LaunchConfiguration("online_training_enabled")),
                package="vision",
                executable="vision_online_trainer",
                name="vision_online_trainer",
                output="screen",
                parameters=[
                    {
                        "checkpoint_dir": LaunchConfiguration("checkpoint_dir"),
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "replay_dir": LaunchConfiguration("replay_dir"),
                        "image_size": LaunchConfiguration("image_size"),
                        "use_mixed_precision": LaunchConfiguration("use_mixed_precision"),
                        "replay_buffer_capacity": LaunchConfiguration("replay_buffer_capacity"),
                        "min_replay_size": LaunchConfiguration("min_replay_size"),
                        "train_batch_size": LaunchConfiguration("train_batch_size"),
                        "train_steps_per_tick": LaunchConfiguration("train_steps_per_tick"),
                        "train_tick_period_sec": LaunchConfiguration("train_tick_period_sec"),
                        "online_learning_rate": LaunchConfiguration("online_learning_rate"),
                        "geometry_loss_weight": LaunchConfiguration("geometry_loss_weight"),
                        "presence_loss_weight": LaunchConfiguration("presence_loss_weight"),
                        "depth_loss_weight": LaunchConfiguration("depth_loss_weight"),
                        "tensorboard_log_dir": LaunchConfiguration("tensorboard_log_dir"),
                        "tensorboard_run_name": LaunchConfiguration("tensorboard_run_name"),
                    }
                ],
            ),
        ]
    )
