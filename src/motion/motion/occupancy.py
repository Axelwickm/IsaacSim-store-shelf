from __future__ import annotations

from typing import Any

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from .arm_config import ArmSideConfig


def is_arm_link(link_name: str, arm_config: ArmSideConfig) -> bool:
    if link_name.startswith("yumi_link_"):
        return link_name.endswith(f"_{arm_config.suffix}")
    return link_name.startswith(arm_config.gripper_prefix)


def reserved_point_distance(
    reserved_point: dict[str, Any] | None,
    frame_id: str,
    xyz: tuple[float, float, float],
) -> float | None:
    if not reserved_point:
        return None
    if reserved_point.get("frame_id") != frame_id:
        return None
    reserved_xyz = reserved_point.get("xyz") or []
    if len(reserved_xyz) != 3:
        return None
    dx = float(xyz[0]) - float(reserved_xyz[0])
    dy = float(xyz[1]) - float(reserved_xyz[1])
    dz = float(xyz[2]) - float(reserved_xyz[2])
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def serialize_planned_trajectory(
    planned_trajectory: Any | None,
    stamp_ns: int,
) -> dict[str, Any] | None:
    if planned_trajectory is None:
        return None
    joint_trajectory = getattr(planned_trajectory, "joint_trajectory", None)
    if joint_trajectory is None:
        return None
    joint_names = list(getattr(joint_trajectory, "joint_names", []))
    points = list(getattr(joint_trajectory, "points", []))
    if not joint_names or not points:
        return None
    serialized_points = []
    final_time = 0.0
    for point in points:
        time_from_start = (
            float(point.time_from_start.sec)
            + float(point.time_from_start.nanosec) * 1e-9
        )
        final_time = max(final_time, time_from_start)
        serialized_points.append(
            {
                "time_from_start": time_from_start,
                "positions": [float(value) for value in point.positions],
            }
        )
    return {
        "joint_names": joint_names,
        "points": serialized_points,
        "published_ns": int(stamp_ns),
        "final_time": final_time,
    }


def sample_peer_plan_joint_positions(
    peer_plan: dict[str, Any] | None,
    now_ns: int,
    extra_horizon_seconds: float,
) -> dict[str, float]:
    if not isinstance(peer_plan, dict):
        return {}
    joint_names = list(peer_plan.get("joint_names") or [])
    points = list(peer_plan.get("points") or [])
    published_ns = int(peer_plan.get("published_ns") or 0)
    final_time = float(peer_plan.get("final_time") or 0.0)
    if not joint_names or not points or published_ns <= 0:
        return {}
    elapsed = max(0.0, (now_ns - published_ns) * 1e-9)
    if elapsed > final_time + extra_horizon_seconds:
        return {}
    sample = sample_peer_plan_point(points, elapsed)
    positions = list(sample.get("positions") or [])
    return {
        joint_name: float(position)
        for joint_name, position in zip(joint_names, positions, strict=False)
    }


def sample_peer_plan_point(
    points: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    if elapsed <= float(points[0].get("time_from_start", 0.0)):
        return points[0]
    for index in range(1, len(points)):
        previous = points[index - 1]
        current = points[index]
        previous_time = float(previous.get("time_from_start", 0.0))
        current_time = float(current.get("time_from_start", 0.0))
        if elapsed > current_time:
            continue
        span = max(current_time - previous_time, 1e-9)
        ratio = min(max((elapsed - previous_time) / span, 0.0), 1.0)
        previous_positions = list(previous.get("positions") or [])
        current_positions = list(current.get("positions") or [])
        return {
            "time_from_start": elapsed,
            "positions": [
                float(previous_position)
                + (float(current_position) - float(previous_position)) * ratio
                for previous_position, current_position in zip(
                    previous_positions,
                    current_positions,
                    strict=False,
                )
            ],
        }
    return points[-1]


def merged_position_map(
    latest_joint_state: JointState | None,
    peer_plan: dict[str, Any] | None,
    now_ns: int,
    extra_horizon_seconds: float,
) -> dict[str, float]:
    if latest_joint_state is None:
        return {}
    position_map = {
        name: float(position)
        for name, position in zip(
            latest_joint_state.name,
            latest_joint_state.position,
            strict=False,
        )
    }
    position_map.update(
        sample_peer_plan_joint_positions(peer_plan, now_ns, extra_horizon_seconds)
    )
    return position_map


def build_other_arm_collision_objects(
    *,
    spheres: Any,
    sphere_links: list[str],
    other_arm_config: ArmSideConfig,
    moveit_target_frame: str,
    radius_padding: float,
    id_prefix: str = "other_arm_sphere",
) -> tuple[list[CollisionObject], set[str]]:
    collision_objects: list[CollisionObject] = []
    other_arm_proxy_links: set[str] = set()
    for index, sphere_values in enumerate(spheres):
        link_name = sphere_links[index]
        if not is_arm_link(link_name, other_arm_config):
            continue
        if link_name in other_arm_config.proxy_excluded_links:
            continue
        other_arm_proxy_links.add(link_name)
        x, y, z, radius = [float(value) for value in sphere_values]
        if radius <= 0.0:
            continue

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [radius + radius_padding]

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0

        collision_object = CollisionObject()
        collision_object.header.frame_id = moveit_target_frame
        collision_object.id = f"{id_prefix}_{index}"
        collision_object.operation = CollisionObject.ADD
        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(pose)
        collision_objects.append(collision_object)

    return collision_objects, other_arm_proxy_links
