#!/usr/bin/env python3

from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from isaac_ros_cumotion_interfaces.srv import PublishStaticPlanningScene
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from rclpy.parameter import Parameter
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
        self.declare_parameter("apply_planning_scene_service", "/apply_planning_scene")
        self.declare_parameter("cumotion_scene_service", "")
        self.declare_parameter(
            "planning_scene_topics",
            Parameter.Type.STRING_ARRAY,
        )
        self._apply_service = str(
            self.get_parameter("apply_planning_scene_service").value
        )
        self._cumotion_scene_service = str(
            self.get_parameter("cumotion_scene_service").value
        ).strip()
        self._planning_scene_topics = [
            str(topic)
            for topic in self.get_parameter("planning_scene_topics").value
            if str(topic)
        ]
        self._client = self.create_client(ApplyPlanningScene, self._apply_service)
        self._planning_scene_publishers = [
            self.create_publisher(PlanningScene, topic, 10)
            for topic in self._planning_scene_topics
        ]
        self._cumotion_scene_server = (
            self.create_service(
                PublishStaticPlanningScene,
                self._cumotion_scene_service,
                self._handle_cumotion_scene_request,
            )
            if self._cumotion_scene_service
            else None
        )
        self._timer = self.create_timer(0.5, self._apply_once)
        self._applied = False
        if self._cumotion_scene_service:
            self.get_logger().info(
                f"Serving cuMotion static planning scene on {self._cumotion_scene_service}"
            )

    def _apply_once(self) -> None:
        if self._applied:
            return
        if not self._client.service_is_ready():
            self.get_logger().info(f"Waiting for {self._apply_service}")
            return

        config_path = Path(str(self.get_parameter("collision_config").value))
        scene = self._build_planning_scene(config_path)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._client.call_async(request)
        future.add_done_callback(self._handle_apply_result)
        self._applied = True
        self._timer.cancel()

    def _handle_cumotion_scene_request(
        self,
        _request: PublishStaticPlanningScene.Request,
        response: PublishStaticPlanningScene.Response,
    ) -> PublishStaticPlanningScene.Response:
        config_path = Path(str(self.get_parameter("collision_config").value))
        scene = self._build_planning_scene(config_path)
        for publisher in self._planning_scene_publishers:
            publisher.publish(scene)

        response.success = True
        response.message = f"Published {len(scene.world.collision_objects)} objects"
        response.planning_scene = scene
        self.get_logger().info(
            f"Published static shelf collision objects for cuMotion on {self._cumotion_scene_service}"
        )
        return response

    def _build_planning_scene(self, config_path: Path) -> PlanningScene:
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = self._load_collision_objects(config_path)
        return scene

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
