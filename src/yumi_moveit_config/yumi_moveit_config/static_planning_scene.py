#!/usr/bin/env python3

from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


class StaticPlanningSceneNode(Node):
    def __init__(self) -> None:
        super().__init__("static_planning_scene")
        self.declare_parameter(
            "collision_config",
            str(
                Path(get_package_share_directory("yumi_moveit_config"))
                / "config"
                / "store_shelf_collision.yaml"
            ),
        )
        self._client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._timer = self.create_timer(0.5, self._apply_once)
        self._applied = False

    def _apply_once(self) -> None:
        if self._applied:
            return
        if not self._client.service_is_ready():
            self.get_logger().info("Waiting for /apply_planning_scene")
            return

        config_path = Path(str(self.get_parameter("collision_config").value))
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = self._load_collision_objects(config_path)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._client.call_async(request)
        future.add_done_callback(self._handle_apply_result)
        self._applied = True
        self._timer.cancel()

    def _load_collision_objects(self, config_path: Path) -> list[CollisionObject]:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        frame_id = str(config["frame_id"])
        collision_objects = []
        for entry in config.get("objects", []):
            if entry.get("type") != "box":
                raise ValueError(f"Unsupported collision object type: {entry!r}")
            center = [float(value) for value in entry["center"]]
            size = [float(value) for value in entry["size"]]

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = size

            pose = Pose()
            pose.position.x = center[0]
            pose.position.y = center[1]
            pose.position.z = center[2]
            pose.orientation.w = 1.0

            collision_object = CollisionObject()
            collision_object.header.frame_id = frame_id
            collision_object.id = str(entry["id"])
            collision_object.operation = CollisionObject.ADD
            collision_object.primitives.append(primitive)
            collision_object.primitive_poses.append(pose)
            collision_objects.append(collision_object)

        return collision_objects

    def _handle_apply_result(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to apply static planning scene: {exc}")
            return
        if response.success:
            self.get_logger().info("Applied static shelf collision objects")
        else:
            self.get_logger().error("MoveIt rejected static shelf collision objects")


def main() -> None:
    rclpy.init()
    node = StaticPlanningSceneNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
