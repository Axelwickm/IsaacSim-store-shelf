#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class CumotionSpherePublisher(Node):
    def __init__(self) -> None:
        super().__init__("cumotion_sphere_publisher")
        self.declare_parameter("robot_xrdf", "/workspace/usd/robot/yumi_isaacsim.xrdf")
        self.declare_parameter("urdf_path", "/workspace/usd/robot/yumi_isaacsim.urdf")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("marker_topic", "/cumotion/robot_spheres")
        self.declare_parameter("frame_id", "yumi_body")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("collision_log_rate_hz", 1.0)
        self.declare_parameter("collision_epsilon", 0.001)

        self._joint_positions: dict[str, float] = {}
        self._last_markers: set[tuple[str, int]] = set()
        self._model = None
        self._tensor_args = None
        self._joint_names: list[str] = []
        self._sphere_links: list[str] = []
        self._ignored_link_pairs: set[frozenset[str]] = set()
        self._last_collision_log_time = self.get_clock().now()

        self._load_model()

        joint_states_topic = (
            self.get_parameter("joint_states_topic").get_parameter_value().string_value
        )
        marker_topic = self.get_parameter("marker_topic").get_parameter_value().string_value
        publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        if publish_rate_hz <= 0.0:
            publish_rate_hz = 10.0

        self._publisher = self.create_publisher(MarkerArray, marker_topic, 10)
        self._marker_topic = marker_topic
        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)
        self.create_timer(1.0 / publish_rate_hz, self._publish_spheres)
        self.get_logger().info(
            f"Publishing cuMotion robot collision spheres on {marker_topic} "
            f"from joint states {joint_states_topic}"
        )

    def _load_xrdf_collision_config(
        self, robot_xrdf: Path
    ) -> tuple[list[str], set[frozenset[str]]]:
        import yaml

        with robot_xrdf.open("r", encoding="utf-8") as stream:
            xrdf: dict[str, Any] = yaml.safe_load(stream)

        collision = xrdf.get("collision") or {}
        geometry_name = collision.get("geometry")
        geometry = ((xrdf.get("geometry") or {}).get(geometry_name) or {})
        sphere_config = geometry.get("spheres") or {}

        sphere_links: list[str] = []
        for link_name, spheres in sphere_config.items():
            sphere_links.extend([str(link_name)] * len(spheres or []))

        self_collision = xrdf.get("self_collision") or {}
        ignored_link_pairs: set[frozenset[str]] = set()
        for link_name, ignored_links in (self_collision.get("ignore") or {}).items():
            for ignored_link in ignored_links or []:
                ignored_link_pairs.add(frozenset((str(link_name), str(ignored_link))))

        return sphere_links, ignored_link_pairs

    def _load_model(self) -> None:
        from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
        from curobo.cuda_robot_model.cuda_robot_model import (
            CudaRobotModel,
            CudaRobotModelConfig,
        )
        from curobo.cuda_robot_model.util import load_robot_yaml
        from curobo.types.base import TensorDeviceType
        from curobo.types.file_path import ContentPath

        robot_xrdf = Path(
            self.get_parameter("robot_xrdf").get_parameter_value().string_value
        )
        urdf_path = Path(self.get_parameter("urdf_path").get_parameter_value().string_value)
        if not robot_xrdf.is_file():
            raise FileNotFoundError(f"cuMotion XRDF not found: {robot_xrdf}")
        if not urdf_path.is_file():
            raise FileNotFoundError(f"cuMotion URDF not found: {urdf_path}")

        self._sphere_links, self._ignored_link_pairs = self._load_xrdf_collision_config(
            robot_xrdf
        )
        self._tensor_args = TensorDeviceType()
        content_path = ContentPath(
            robot_xrdf_absolute_path=str(robot_xrdf),
            robot_urdf_absolute_path=str(urdf_path),
        )
        robot_yaml = load_robot_yaml(content_path)
        kinematics_config = robot_yaml["robot_cfg"]["kinematics"]
        model_config = CudaRobotModelConfig.from_config(
            CudaRobotGeneratorConfig(
                **kinematics_config,
                tensor_args=self._tensor_args,
            )
        )
        self._model = CudaRobotModel(model_config)
        self._joint_names = list(self._model.joint_names)
        self.get_logger().info(
            "Loaded cuMotion sphere model with active joints: "
            + ", ".join(self._joint_names)
        )
        self.get_logger().info(
            f"Loaded {len(self._sphere_links)} XRDF collision spheres and "
            f"{len(self._ignored_link_pairs)} ignored self-collision link pairs"
        )

    def _on_joint_state(self, message: JointState) -> None:
        for name, position in zip(message.name, message.position, strict=False):
            self._joint_positions[name] = float(position)

    def _publish_spheres(self) -> None:
        if self._model is None or self._tensor_args is None:
            return
        if not self._joint_positions:
            return
        if any(name not in self._joint_positions for name in self._joint_names):
            missing = [
                name for name in self._joint_names if name not in self._joint_positions
            ]
            self.get_logger().debug(
                "Waiting for joint states for cuMotion joints: " + ", ".join(missing)
            )
            return

        import torch

        positions = [
            self._joint_positions[name]
            for name in self._joint_names
        ]
        q = torch.tensor(
            [positions],
            device=self._tensor_args.device,
            dtype=self._tensor_args.dtype,
        )
        spheres = self._model.get_state(q).link_spheres_tensor[0].detach().cpu().numpy()
        if len(spheres) != len(self._sphere_links):
            self.get_logger().warn(
                f"cuMotion returned {len(spheres)} spheres but XRDF declares "
                f"{len(self._sphere_links)} spheres; collision labels may be offset.",
                throttle_duration_sec=5.0,
            )
        self._log_self_collisions(spheres)

        frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        current_markers: set[tuple[str, int]] = set()

        for index, sphere in enumerate(spheres):
            x, y, z, radius = [float(value) for value in sphere]
            if radius <= 0.0:
                continue
            link_name = self._sphere_link_name(index)
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = f"cumotion_{link_name}"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0
            marker.scale.x = radius * 2.0
            marker.scale.y = radius * 2.0
            marker.scale.z = radius * 2.0
            marker.color = self._link_color(link_name)
            marker.text = link_name
            markers.markers.append(marker)
            current_markers.add((marker.ns, marker.id))

        for namespace, marker_id in self._last_markers - current_markers:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = namespace
            marker.id = marker_id
            marker.action = Marker.DELETE
            markers.markers.append(marker)

        self._last_markers = current_markers
        self._publisher.publish(markers)
        if self._publisher.get_subscription_count() == 0:
            self.get_logger().warn(
                f"No subscribers for {self._marker_topic}; add a MarkerArray display "
                "in RViz for the cuMotion spheres.",
                throttle_duration_sec=5.0,
            )

    def _sphere_link_name(self, sphere_index: int) -> str:
        if sphere_index < len(self._sphere_links):
            return self._sphere_links[sphere_index]
        return "unknown"

    def _link_color(self, link_name: str) -> ColorRGBA:
        palette = [
            ColorRGBA(r=1.0, g=0.15, b=0.05, a=0.35),
            ColorRGBA(r=0.1, g=0.55, b=1.0, a=0.35),
            ColorRGBA(r=0.05, g=0.85, b=0.25, a=0.35),
            ColorRGBA(r=1.0, g=0.75, b=0.05, a=0.35),
            ColorRGBA(r=0.8, g=0.25, b=1.0, a=0.35),
            ColorRGBA(r=0.0, g=0.85, b=0.85, a=0.35),
        ]
        color_index = sum(ord(character) for character in link_name) % len(palette)
        return palette[color_index]

    def _log_self_collisions(self, spheres) -> None:
        log_rate_hz = (
            self.get_parameter("collision_log_rate_hz").get_parameter_value().double_value
        )
        if log_rate_hz <= 0.0:
            return

        now = self.get_clock().now()
        elapsed = (now - self._last_collision_log_time).nanoseconds / 1e9
        if elapsed < 1.0 / log_rate_hz:
            return
        self._last_collision_log_time = now

        epsilon = (
            self.get_parameter("collision_epsilon").get_parameter_value().double_value
        )
        collisions: list[tuple[float, str, int, str, int, float, float]] = []

        active = []
        for index, sphere in enumerate(spheres):
            x, y, z, radius = [float(value) for value in sphere]
            if radius > 0.0:
                active.append((index, self._sphere_link_name(index), x, y, z, radius))

        for first_offset, first in enumerate(active):
            first_index, first_link, first_x, first_y, first_z, first_radius = first
            for second in active[first_offset + 1 :]:
                (
                    second_index,
                    second_link,
                    second_x,
                    second_y,
                    second_z,
                    second_radius,
                ) = second
                if first_link == second_link:
                    continue
                if frozenset((first_link, second_link)) in self._ignored_link_pairs:
                    continue

                dx = first_x - second_x
                dy = first_y - second_y
                dz = first_z - second_z
                distance = (dx * dx + dy * dy + dz * dz) ** 0.5
                radius_sum = first_radius + second_radius
                penetration = radius_sum - distance
                if penetration > epsilon:
                    collisions.append(
                        (
                            penetration,
                            first_link,
                            first_index,
                            second_link,
                            second_index,
                            distance,
                            radius_sum,
                        )
                    )

        if not collisions:
            self.get_logger().info(
                f"cuMotion sphere self-collision check: no non-ignored overlaps "
                f"among {len(active)} active spheres"
            )
            return

        collisions.sort(reverse=True)
        details = "; ".join(
            f"{first_link}[{first_index}] <-> {second_link}[{second_index}] "
            f"penetration={penetration:.4f}m distance={distance:.4f}m "
            f"radius_sum={radius_sum:.4f}m"
            for (
                penetration,
                first_link,
                first_index,
                second_link,
                second_index,
                distance,
                radius_sum,
            ) in collisions[:8]
        )
        self.get_logger().warn(
            f"cuMotion sphere self-collision check: {len(collisions)} "
            f"non-ignored overlaps. Worst: {details}"
        )


def main() -> None:
    rclpy.init()
    node = CumotionSpherePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
