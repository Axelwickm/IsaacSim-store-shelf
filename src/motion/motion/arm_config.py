from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmSideConfig:
    side: str
    suffix: str
    joint_order: tuple[str, ...]
    move_group_name: str
    end_effector_link: str
    direct_trajectory_action: str
    gripper_joint: str
    gripper_mimic_joint: str
    gripper_prefix: str
    proxy_excluded_links: frozenset[str]

    @property
    def planning_start_joints(self) -> tuple[str, ...]:
        return self.joint_order + (self.gripper_joint,)


_ARM_JOINT_LIMIT_BOUNDS = {
    1: (-2.84, 2.84),
    2: (-2.4, 0.65),
    7: (-2.84, 2.84),
    3: (-2.0, 1.29),
    4: (-4.9, 4.9),
    5: (-1.43, 2.3),
    6: (-3.89, 3.89),
}


JOINT_POSITION_LIMITS = {
    f"yumi_joint_{joint_index}_{arm_suffix}": limits
    for arm_suffix in ("l", "r")
    for joint_index, limits in _ARM_JOINT_LIMIT_BOUNDS.items()
}


JOINT_LIMIT_MARGIN_RAD = 1e-3


ARM_SIDE_CONFIGS: dict[str, ArmSideConfig] = {
    "left": ArmSideConfig(
        side="left",
        suffix="l",
        joint_order=(
            "yumi_joint_1_l",
            "yumi_joint_2_l",
            "yumi_joint_7_l",
            "yumi_joint_3_l",
            "yumi_joint_4_l",
            "yumi_joint_5_l",
            "yumi_joint_6_l",
        ),
        move_group_name="left_arm",
        end_effector_link="gripper_l_grasp_frame",
        direct_trajectory_action="left_arm_controller/follow_joint_trajectory",
        gripper_joint="gripper_l_joint",
        gripper_mimic_joint="gripper_l_joint_m",
        gripper_prefix="gripper_l_",
        proxy_excluded_links=frozenset({"yumi_link_1_l", "yumi_link_2_l"}),
    ),
    "right": ArmSideConfig(
        side="right",
        suffix="r",
        joint_order=(
            "yumi_joint_1_r",
            "yumi_joint_2_r",
            "yumi_joint_7_r",
            "yumi_joint_3_r",
            "yumi_joint_4_r",
            "yumi_joint_5_r",
            "yumi_joint_6_r",
        ),
        move_group_name="right_arm",
        end_effector_link="gripper_r_grasp_frame",
        direct_trajectory_action="right_arm_controller/follow_joint_trajectory",
        gripper_joint="gripper_r_joint",
        gripper_mimic_joint="gripper_r_joint_m",
        gripper_prefix="gripper_r_",
        proxy_excluded_links=frozenset({"yumi_link_1_r", "yumi_link_2_r"}),
    ),
}


def get_arm_side_config(side: str) -> ArmSideConfig:
    key = side.strip().lower()
    if key not in ARM_SIDE_CONFIGS:
        raise ValueError(
            "arm side must be one of: " + ", ".join(sorted(ARM_SIDE_CONFIGS))
        )
    return ARM_SIDE_CONFIGS[key]
