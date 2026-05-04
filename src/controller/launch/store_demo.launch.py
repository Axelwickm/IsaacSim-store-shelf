from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    sim_launch = Path(get_package_share_directory("controller")) / "launch" / "sim.launch.py"
    vision_launch = Path(get_package_share_directory("vision")) / "launch" / "inference.launch.py"
    motion_launch = Path(get_package_share_directory("motion")) / "launch" / "motion.launch.py"

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
            DeclareLaunchArgument(
                "vision_online_training_enabled",
                default_value="false",
                description="Record replay samples and launch the external online vision trainer.",
            ),
            DeclareLaunchArgument(
                "vision_replay_dir",
                default_value="/workspace/replay/vision",
                description="Directory used as the durable replay-sample queue.",
            ),
            DeclareLaunchArgument(
                "vision_checkpoint_reload_period_sec",
                default_value="5.0",
                description="Seconds between inference checkpoint hot-reload checks.",
            ),
            DeclareLaunchArgument(
                "vision_exploration_epsilon",
                default_value="0.10",
                description="Exploration probability for arm/query selection.",
            ),
            DeclareLaunchArgument(
                "vision_camera_frame_convention",
                default_value="ros_optical",
                description="Vision camera TF convention: usd_camera or ros_optical.",
            ),
            DeclareLaunchArgument(
                "vision_ground_truth_items_topic",
                default_value="/vision/ground_truth_items",
                description="IsaacSim-published item world centers used for green debug markers.",
            ),
            DeclareLaunchArgument(
                "vision_suggested_item_markers_topic",
                default_value="/vision/suggested_item_markers",
                description="RViz MarkerArray topic for multiple projected suggested 3D item points.",
            ),
            DeclareLaunchArgument(
                "vision_max_suggested_markers",
                default_value="32",
                description="Maximum projected suggested 3D item markers to publish.",
            ),
            DeclareLaunchArgument(
                "vision_replay_buffer_capacity",
                default_value="2048",
                description="Maximum replay samples kept in memory for online value learning.",
            ),
            DeclareLaunchArgument(
                "vision_min_replay_size",
                default_value="2",
                description="Minimum replay samples before online training begins.",
            ),
            DeclareLaunchArgument(
                "vision_train_batch_size",
                default_value="16",
                description="Online replay training batch size.",
            ),
            DeclareLaunchArgument(
                "vision_train_steps_per_tick",
                default_value="4",
                description="Maximum online optimizer steps to run per timer tick.",
            ),
            DeclareLaunchArgument(
                "vision_train_tick_period_sec",
                default_value="0.01",
                description="Seconds between online optimizer timer ticks.",
            ),
            DeclareLaunchArgument(
                "vision_online_learning_rate",
                default_value="1e-3",
                description="Learning rate for online replay training.",
            ),
            DeclareLaunchArgument(
                "vision_geometry_loss_weight",
                default_value="1.0",
                description="Weight for online GT-projected query-center supervision.",
            ),
            DeclareLaunchArgument(
                "vision_presence_loss_weight",
                default_value="0.2",
                description="Weight for online GT query-presence supervision.",
            ),
            DeclareLaunchArgument(
                "vision_depth_loss_weight",
                default_value="0.2",
                description="Weight for online GT projected-depth supervision.",
            ),
            DeclareLaunchArgument(
                "vision_tensorboard_log_dir",
                default_value="/workspace/tensorboard/vision",
                description="TensorBoard base log directory used only when online vision training is enabled.",
            ),
            DeclareLaunchArgument(
                "vision_tensorboard_run_name",
                default_value="",
                description=(
                    "Optional TensorBoard run subdirectory name. If empty, vision "
                    "generates a unique timestamped name or reuses the checkpoint run."
                ),
            ),
            DeclareLaunchArgument(
                "motion_pipeline_id",
                default_value="isaac_ros_cumotion",
                description="MoveIt planning pipeline requested by the motion planner.",
            ),
            DeclareLaunchArgument(
                "motion_planner_id",
                default_value="cuMotion",
                description="Planner id requested by the motion planner.",
            ),
            DeclareLaunchArgument(
                "controller_spawner_delay",
                default_value="30.0",
                description="Seconds to wait before activating ros2_control controllers.",
            ),
            DeclareLaunchArgument(
                "move_group_delay",
                default_value="5.0",
                description=(
                    "Seconds to wait after controller spawners finish before starting "
                    "move_group."
                ),
            ),
            DeclareLaunchArgument(
                "move_group_log_level",
                default_value="info",
                description="ROS log level for move_group.",
            ),
            DeclareLaunchArgument(
                "use_moveit_rviz",
                default_value="false",
                description="Whether to launch RViz for MoveIt and cuMotion debug markers.",
            ),
            DeclareLaunchArgument(
                "right_cumotion_robot_xrdf",
                default_value="/workspace/usd/robot/yumi_isaacsim_right_arm.xrdf",
                description="XRDF file for the right-arm standalone cuMotion planner node.",
            ),
            DeclareLaunchArgument(
                "right_cumotion_urdf_path",
                default_value="/workspace/usd/robot/yumi_isaacsim_right_arm.urdf",
                description="URDF file for the right-arm standalone cuMotion planner node.",
            ),
            DeclareLaunchArgument(
                "left_cumotion_robot_xrdf",
                default_value="/workspace/usd/robot/yumi_isaacsim_left_arm.xrdf",
                description="XRDF file for the left-arm standalone cuMotion planner node.",
            ),
            DeclareLaunchArgument(
                "left_cumotion_urdf_path",
                default_value="/workspace/usd/robot/yumi_isaacsim_left_arm.urdf",
                description="URDF file for the left-arm standalone cuMotion planner node.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(sim_launch)),
                launch_arguments={
                    "configuration": "store_demo",
                    "use_moveit": "true",
                    "planning_pipeline": LaunchConfiguration("motion_pipeline_id"),
                    "controller_spawner_delay": LaunchConfiguration(
                        "controller_spawner_delay"
                    ),
                    "move_group_delay": LaunchConfiguration("move_group_delay"),
                    "move_group_log_level": LaunchConfiguration("move_group_log_level"),
                    "use_moveit_rviz": LaunchConfiguration("use_moveit_rviz"),
                    "right_planning_cumotion_robot_xrdf": LaunchConfiguration(
                        "right_cumotion_robot_xrdf"
                    ),
                    "right_planning_cumotion_urdf_path": LaunchConfiguration(
                        "right_cumotion_urdf_path"
                    ),
                    "left_planning_cumotion_robot_xrdf": LaunchConfiguration(
                        "left_cumotion_robot_xrdf"
                    ),
                    "left_planning_cumotion_urdf_path": LaunchConfiguration(
                        "left_cumotion_urdf_path"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(vision_launch)),
                launch_arguments={
                    "checkpoint_dir": LaunchConfiguration("vision_checkpoint_dir"),
                    "checkpoint_path": LaunchConfiguration("vision_checkpoint_path"),
                    "image_topic": "/camera/image_raw",
                    "image_size": LaunchConfiguration("vision_image_size"),
                    "ground_truth_items_topic": LaunchConfiguration(
                        "vision_ground_truth_items_topic"
                    ),
                    "suggested_item_markers_topic": LaunchConfiguration(
                        "vision_suggested_item_markers_topic"
                    ),
                    "online_training_enabled": LaunchConfiguration("vision_online_training_enabled"),
                    "replay_dir": LaunchConfiguration("vision_replay_dir"),
                    "checkpoint_reload_period_sec": LaunchConfiguration(
                        "vision_checkpoint_reload_period_sec"
                    ),
                    "exploration_epsilon": LaunchConfiguration("vision_exploration_epsilon"),
                    "query_presence_threshold": "0.20",
                    "max_suggested_markers": LaunchConfiguration("vision_max_suggested_markers"),
                    "camera_frame_convention": LaunchConfiguration("vision_camera_frame_convention"),
                    "replay_buffer_capacity": LaunchConfiguration("vision_replay_buffer_capacity"),
                    "min_replay_size": LaunchConfiguration("vision_min_replay_size"),
                    "train_batch_size": LaunchConfiguration("vision_train_batch_size"),
                    "train_steps_per_tick": LaunchConfiguration("vision_train_steps_per_tick"),
                    "train_tick_period_sec": LaunchConfiguration("vision_train_tick_period_sec"),
                    "online_learning_rate": LaunchConfiguration("vision_online_learning_rate"),
                    "geometry_loss_weight": LaunchConfiguration("vision_geometry_loss_weight"),
                    "presence_loss_weight": LaunchConfiguration("vision_presence_loss_weight"),
                    "depth_loss_weight": LaunchConfiguration("vision_depth_loss_weight"),
                    "tensorboard_log_dir": LaunchConfiguration("vision_tensorboard_log_dir"),
                    "tensorboard_run_name": LaunchConfiguration("vision_tensorboard_run_name"),
                    "use_mixed_precision": LaunchConfiguration(
                        "vision_use_mixed_precision"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(motion_launch)),
                launch_arguments={
                    "pipeline_id": LaunchConfiguration("motion_pipeline_id"),
                    "planner_id": LaunchConfiguration("motion_planner_id"),
                    "right_cumotion_robot_xrdf": LaunchConfiguration(
                        "right_cumotion_robot_xrdf"
                    ),
                    "right_cumotion_urdf_path": LaunchConfiguration(
                        "right_cumotion_urdf_path"
                    ),
                    "left_cumotion_robot_xrdf": LaunchConfiguration(
                        "left_cumotion_robot_xrdf"
                    ),
                    "left_cumotion_urdf_path": LaunchConfiguration(
                        "left_cumotion_urdf_path"
                    ),
                    "launch_left_planner": "true",
                }.items(),
            ),
        ]
    )
