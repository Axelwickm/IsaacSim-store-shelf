from pathlib import Path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    sim_launch = Path(get_package_share_directory("controller")) / "launch" / "sim.launch.py"
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(sim_launch)),
                launch_arguments={
                    "configuration": "store_demo",
                    "use_moveit": "true",
                    "planning_pipeline": "ompl",
                }.items(),
            )
        ]
    )
