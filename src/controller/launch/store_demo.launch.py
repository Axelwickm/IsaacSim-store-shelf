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
                "vision_collect_training_data",
                default_value="false",
                description="Record supervised Replicator samples and planner feedback into replay.",
            ),
            DeclareLaunchArgument(
                "vision_replay_dir",
                default_value="/workspace/replay",
                description="Directory used as the durable replay-sample queue.",
            ),
            DeclareLaunchArgument(
                "vision_checkpoint_reload_period_sec",
                default_value="20.0",
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
                "vision_cuda_memory_log_period_sec",
                default_value="30.0",
                description="Seconds between vision CUDA memory accounting logs.",
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
                    "store_demo_capture_training_data": LaunchConfiguration(
                        "vision_collect_training_data"
                    ),
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
                    "collect_training_data": LaunchConfiguration("vision_collect_training_data"),
                    "replay_dir": LaunchConfiguration("vision_replay_dir"),
                    "checkpoint_reload_period_sec": LaunchConfiguration(
                        "vision_checkpoint_reload_period_sec"
                    ),
                    "exploration_epsilon": LaunchConfiguration("vision_exploration_epsilon"),
                    "query_presence_threshold": "0.20",
                    "max_suggested_markers": LaunchConfiguration("vision_max_suggested_markers"),
                    "camera_frame_convention": LaunchConfiguration("vision_camera_frame_convention"),
                    "use_mixed_precision": LaunchConfiguration(
                        "vision_use_mixed_precision"
                    ),
                    "cuda_memory_log_period_sec": LaunchConfiguration(
                        "vision_cuda_memory_log_period_sec"
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
