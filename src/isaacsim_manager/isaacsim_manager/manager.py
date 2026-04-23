#!/usr/bin/env python3

import os
import json
import traceback
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from isaacsim_manager.image_collection import (
    DEFAULT_COLLECT_VISION_SCENE,
    collect_vision_data,
    static,
    update_simulation_app,
)

START_HEADED = "headed"
START_HEADLESS = "headless"
CONTROL_TOPIC = "isaacsim_manager/control"
ACK_TOPIC = "isaacsim_manager/ack"
CONFIGURATION_FUNCTIONS = {
    "collect_vision_data": collect_vision_data,
    "static": static,
}


class IsaacSimManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("isaacsim_manager")
        self._last_start_mode = START_HEADED
        self._simulation_app = None
        self._active_configuration = ""
        self._isaacsim_root = Path(
            os.environ.get("ISAACSIM_ROOT", "/isaac-sim")
        ).resolve()
        self._control_subscription = self.create_subscription(
            String,
            CONTROL_TOPIC,
            self._handle_control,
            10,
        )
        self._ack_publisher = self.create_publisher(String, ACK_TOPIC, 10)
        self._app_update_timer = self.create_timer(
            1.0 / 60.0,
            self._update_simulation_app,
        )
        self.get_logger().info(f"Isaac Sim manager loaded from {__file__}")
        self.get_logger().info(f"Listening for manager commands on /{CONTROL_TOPIC}")
        self.get_logger().info(f"Publishing manager acknowledgements on /{ACK_TOPIC}")
        self.get_logger().info("Isaac Sim manager service ready")

    def _update_simulation_app(self) -> None:
        if self._simulation_app is None:
            return
        update_simulation_app(self._simulation_app)

    def _is_running(self) -> bool:
        return self._simulation_app is not None

    def _create_simulation_app(self, mode: str):
        from isaacsim import SimulationApp

        headless = mode == START_HEADLESS
        extra_args = [
            "--portable-root",
            str(self._isaacsim_root / ".local/share/ov/data"),
        ]
        return SimulationApp({"headless": headless, "extra_args": extra_args})

    def _start_isaacsim(self, mode: str, configuration: str) -> tuple[bool, str]:
        if self._is_running():
            return False, "Isaac Sim is already running"

        try:
            self.get_logger().info(
                f"Creating SimulationApp headless={mode == START_HEADLESS}"
            )
            self._simulation_app = self._create_simulation_app(mode)
            result = CONFIGURATION_FUNCTIONS[configuration](self._simulation_app)
        except Exception as exc:
            self.get_logger().error(
                "Failed while starting simulation configuration:\n"
                + traceback.format_exc()
            )
            self._close_simulation_app()
            return False, f"Failed to start Isaac Sim: {exc}"

        self._last_start_mode = mode
        self._active_configuration = configuration
        return True, f"Isaac Sim started in {mode} mode: {result}"

    def _close_simulation_app(self) -> None:
        simulation_app = self._simulation_app
        self._simulation_app = None
        self._active_configuration = ""
        if simulation_app is None:
            return
        try:
            simulation_app.close()
        except Exception as exc:
            self.get_logger().warning(f"Failed while closing SimulationApp: {exc}")

    def _stop_isaacsim(self) -> tuple[bool, str]:
        if not self._is_running():
            return False, "Isaac Sim is not running"

        self._close_simulation_app()
        return True, "Isaac Sim stopped"

    def _handle_restart(self) -> tuple[bool, str]:
        configuration = self._active_configuration
        if self._is_running():
            stopped, stop_message = self._stop_isaacsim()
            if not stopped:
                return False, stop_message

        if not configuration:
            return False, "No active simulation configuration to restart"
        success, message = self._start_isaacsim(self._last_start_mode, configuration)
        if success:
            message = f"Isaac Sim restarted in {self._last_start_mode} mode"
        return success, message

    def _parse_control_message(self, message: String) -> tuple[str, str, str]:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return uuid.uuid4().hex, message.data.strip().lower(), ""

        request_id = str(payload.get("id") or uuid.uuid4().hex)
        command = str(payload.get("command") or "").strip().lower()
        configuration = str(payload.get("configuration") or "").strip()
        return request_id, command, configuration

    def _publish_ack(
        self,
        request_id: str,
        success: bool,
        result: str,
        configuration: str,
        scene: str,
    ) -> None:
        ack = String()
        ack.data = json.dumps(
            {
                "id": request_id,
                "success": success,
                "message": result,
                "configuration": configuration,
                "scene": scene,
            },
            separators=(",", ":"),
        )
        self._ack_publisher.publish(ack)
        self.get_logger().info(f"Published manager ack: {ack.data}")

    def _handle_control(self, message: String) -> None:
        request_id, command, configuration = self._parse_control_message(message)
        scene = os.environ.get(
            "ISAACSIM_COLLECT_VISION_SCENE",
            DEFAULT_COLLECT_VISION_SCENE,
        )
        self.get_logger().info(
            f"Received manager command id={request_id!r} command={command!r} "
            f"configuration={configuration!r}"
        )
        if command in {"start", "start_headless", "restart"} and not configuration:
            success = False
            result = "A simulation configuration is required"
            self.get_logger().warning(result)
            self._publish_ack(request_id, success, result, configuration, scene)
            return

        if configuration and configuration not in CONFIGURATION_FUNCTIONS:
            success = False
            result = f"Unsupported simulation configuration: {configuration!r}"
            self.get_logger().warning(result)
            self._publish_ack(request_id, success, result, configuration, scene)
            return

        if command == "start":
            success, result = self._start_isaacsim(START_HEADED, configuration)
        elif command == "start_headless":
            success, result = self._start_isaacsim(START_HEADLESS, configuration)
        elif command in {"stop", "kill"}:
            success, result = self._stop_isaacsim()
        elif command == "restart":
            success, result = self._handle_restart()
        else:
            success = False
            result = f"Unsupported manager command: {message.data!r}"
            self.get_logger().warning(result)
            self._publish_ack(request_id, success, result, configuration, scene)
            return

        if success:
            self.get_logger().info(result)
        else:
            self.get_logger().warning(result)
        self._publish_ack(request_id, success, result, configuration, scene)


def main() -> None:
    rclpy.init()
    node = IsaacSimManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node._stop_isaacsim()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
