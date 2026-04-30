#!/usr/bin/env python3

from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from isaac_ros_cumotion_interfaces.srv import PublishStaticPlanningScene
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


def load_collision_objects(config_path: Path) -> list[CollisionObject]:
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


class StaticPlanningSceneServer(Node):
    def __init__(self) -> None:
        super().__init__("static_planning_scene_server")
        self.declare_parameter(
            "collision_config",
            str(
                Path(get_package_share_directory("yumi_moveit_config"))
                / "config"
                / "store_shelf_collision.yaml"
            ),
        )
        self.declare_parameter("service_name", "/publish_static_planning_scene")
        self.declare_parameter("service_names", [])
        self.declare_parameter(
            "planning_scene_topics",
            [
                "/planning_scene",
                "/moveit_left/planning_scene",
                "/moveit_right/planning_scene",
            ],
        )

        config_path = Path(str(self.get_parameter("collision_config").value))
        self._collision_objects = load_collision_objects(config_path)
        self._planning_scene_topics = [
            str(topic)
            for topic in self.get_parameter("planning_scene_topics").value
            if str(topic)
        ]
        self._publishers = [
            self.create_publisher(PlanningScene, topic, 10)
            for topic in self._planning_scene_topics
        ]
        service_names = [
            str(name)
            for name in self.get_parameter("service_names").value
            if str(name)
        ]
        if not service_names:
            service_names = [str(self.get_parameter("service_name").value)]
        self._service_names = tuple(dict.fromkeys(service_names))
        self._services = [
            self.create_service(
                PublishStaticPlanningScene,
                service_name,
                self._handle_trigger,
            )
            for service_name in self._service_names
        ]

        self.get_logger().info(
            "Static Planning Scene Server initialized on "
            + ", ".join(self._service_names)
        )

    def _handle_trigger(
        self,
        _request: PublishStaticPlanningScene.Request,
        response: PublishStaticPlanningScene.Response,
    ) -> PublishStaticPlanningScene.Response:
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = list(self._collision_objects)
        for publisher in self._publishers:
            publisher.publish(scene)

        self.get_logger().info(
            f"Published {len(self._collision_objects)} collision objects to "
            + ", ".join(self._planning_scene_topics)
        )
        response.success = True
        response.message = f"Published {len(self._collision_objects)} collision objects"
        response.planning_scene = scene
        return response


def main() -> None:
    rclpy.init()
    node = StaticPlanningSceneServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
