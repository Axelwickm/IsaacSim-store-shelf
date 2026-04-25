#!/usr/bin/env python3

import os
import json
import argparse
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from isaacsim_manager.image_collection import (
    DEFAULT_COLLECT_VISION_SCENE,
    collect_vision_data,
    store_demo,
    static,
    update_simulation_app,
)

START_HEADED = "headed"
START_HEADLESS = "headless"
CONTROL_TOPIC = "isaacsim_manager/control"
ACK_TOPIC = "isaacsim_manager/ack"
CONFIGURATION_NAMES = {"collect_vision_data", "static", "store_demo"}
STARTUP_GRACE_PERIOD_SECONDS = 2.0
CONFIGURATION_FUNCTIONS = {
    "collect_vision_data": collect_vision_data,
    "static": static,
    "store_demo": store_demo,
}


class IsaacSimManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("isaacsim_manager")
        self._last_start_mode = START_HEADED
        self._simulation_process: subprocess.Popen[str] | None = None
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
        simulation_process = self._simulation_process
        if simulation_process is None:
            return
        return_code = simulation_process.poll()
        if return_code is None:
            return

        configuration = self._active_configuration
        self._simulation_process = None
        self._active_configuration = ""
        if return_code == 0:
            self.get_logger().info(
                f"Isaac Sim runner exited cleanly for configuration={configuration!r}"
            )
        else:
            self.get_logger().warning(
                f"Isaac Sim runner exited unexpectedly with code {return_code} "
                f"for configuration={configuration!r}"
            )

    def _is_running(self) -> bool:
        simulation_process = self._simulation_process
        return simulation_process is not None and simulation_process.poll() is None

    def _start_isaacsim(self, mode: str, configuration: str) -> tuple[bool, str]:
        if self._is_running():
            return False, "Isaac Sim is already running"

        try:
            self.get_logger().info(
                f"Starting Isaac Sim runner mode={mode!r} configuration={configuration!r}"
            )
            self._simulation_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "isaacsim_manager.manager",
                    "--runner-mode",
                    mode,
                    "--configuration",
                    configuration,
                ],
                text=True,
            )
            deadline = time.monotonic() + STARTUP_GRACE_PERIOD_SECONDS
            while time.monotonic() < deadline:
                return_code = self._simulation_process.poll()
                if return_code is not None:
                    self._close_simulation_process()
                    return (
                        False,
                        "Isaac Sim runner exited during startup "
                        f"with code {return_code}",
                    )
                time.sleep(0.1)
        except Exception as exc:
            self.get_logger().error(
                "Failed while starting simulation configuration:\n"
                + traceback.format_exc()
            )
            self._close_simulation_process()
            return False, f"Failed to start Isaac Sim: {exc}"

        self._last_start_mode = mode
        self._active_configuration = configuration
        return True, f"Isaac Sim runner started in {mode} mode"

    def _close_simulation_process(self) -> None:
        simulation_process = self._simulation_process
        self._simulation_process = None
        self._active_configuration = ""
        if simulation_process is None:
            return
        try:
            if simulation_process.poll() is None:
                simulation_process.terminate()
                simulation_process.wait(timeout=20.0)
        except Exception as exc:
            self.get_logger().warning(f"Failed while closing Isaac Sim runner: {exc}")
            try:
                if simulation_process.poll() is None:
                    simulation_process.kill()
            except Exception as kill_exc:
                self.get_logger().warning(
                    f"Failed while force-killing Isaac Sim runner: {kill_exc}"
                )

    def _stop_isaacsim(self) -> tuple[bool, str]:
        if not self._is_running():
            return False, "Isaac Sim is not running"

        self._close_simulation_process()
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

        if configuration and configuration not in CONFIGURATION_NAMES:
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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runner-mode", choices=[START_HEADED, START_HEADLESS])
    parser.add_argument("--configuration")
    args, _unknown = parser.parse_known_args()

    if args.runner_mode:
        _run_simulation_app(args.runner_mode, args.configuration)
        return

    rclpy.init()
    node = IsaacSimManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node._stop_isaacsim()
        node.destroy_node()
        rclpy.shutdown()


def _run_simulation_app(mode: str, configuration: str | None) -> None:
    if configuration not in CONFIGURATION_FUNCTIONS:
        raise RuntimeError(f"Unsupported simulation configuration: {configuration!r}")

    from isaacsim import SimulationApp

    enable_ros2_bridge = (
        os.environ.get("ISAACSIM_ENABLE_ROS2_BRIDGE", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    extra_args = ["--portable-root", "/isaac-sim/.local/share/ov/data"]
    if enable_ros2_bridge:
        extra_args.extend(
            [
                "--enable",
                "isaacsim.ros2.bridge",
                "--enable",
                "isaacsim.core.nodes",
                "--enable",
                "omni.syntheticdata",
                "--enable",
                "omni.replicator.core",
            ]
        )
        print(
            "[isaacsim_manager] Enabling Isaac Sim ROS 2 bridge and core nodes "
            "extensions (isaacsim.ros2.bridge, isaacsim.core.nodes, "
            "omni.syntheticdata, omni.replicator.core)",
            flush=True,
        )
    else:
        print(
            "[isaacsim_manager] Isaac Sim ROS 2 bridge extension disabled by "
            "ISAACSIM_ENABLE_ROS2_BRIDGE",
            flush=True,
        )
    os.environ["ISAACSIM_RUNNER_MODE"] = mode
    simulation_app = SimulationApp(
        {"headless": mode == START_HEADLESS, "extra_args": extra_args}
    )
    try:
        try:
            CONFIGURATION_FUNCTIONS[configuration](simulation_app)
        except Exception:
            print(
                "[isaacsim_manager] Simulation configuration failed:\n"
                + traceback.format_exc(),
                flush=True,
            )
            raise
        while simulation_app.is_running():
            update_simulation_app(simulation_app)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
