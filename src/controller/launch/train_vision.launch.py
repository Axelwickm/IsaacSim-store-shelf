from launch import LaunchDescription
from launch.actions import LogInfo


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            LogInfo(
                msg=(
                    "train_vision is a stub for now; it does not start Isaac Sim "
                    "or forward a configuration to the manager."
                )
            ),
        ]
    )
