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
                "collect_training_data",
                default_value="false",
                description="Record planner feedback replay samples for later training.",
            ),
            DeclareLaunchArgument(
                "replay_dir",
                default_value="/workspace/replay",
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
                "use_mixed_precision",
                default_value="true",
                description="Whether to enable CUDA mixed precision.",
            ),
            DeclareLaunchArgument(
                "cuda_memory_log_period_sec",
                default_value="30.0",
                description="Seconds between vision CUDA memory accounting logs.",
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
                        "collect_training_data": LaunchConfiguration("collect_training_data"),
                        "replay_dir": LaunchConfiguration("replay_dir"),
                        "checkpoint_reload_period_sec": LaunchConfiguration(
                            "checkpoint_reload_period_sec"
                        ),
                        "exploration_epsilon": LaunchConfiguration("exploration_epsilon"),
                        "query_presence_threshold": LaunchConfiguration("query_presence_threshold"),
                        "max_suggested_markers": LaunchConfiguration("max_suggested_markers"),
                        "camera_frame_convention": LaunchConfiguration("camera_frame_convention"),
                        "image_size": LaunchConfiguration("image_size"),
                        "use_mixed_precision": LaunchConfiguration("use_mixed_precision"),
                        "cuda_memory_log_period_sec": LaunchConfiguration(
                            "cuda_memory_log_period_sec"
                        ),
                    }
                ],
            ),
        ]
    )
