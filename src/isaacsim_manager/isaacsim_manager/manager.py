#!/usr/bin/env python3

import os
import json
import signal
import subprocess
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


START_HEADED = "headed"
START_HEADLESS = "headless"
CONTROL_TOPIC = "isaacsim_manager/control"
ACK_TOPIC = "isaacsim_manager/ack"


class IsaacSimManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("isaacsim_manager")
        self._last_start_mode = START_HEADED
        self._isaacsim_process: subprocess.Popen[bytes] | None = None
        self._isaacsim_root = Path(
            os.environ.get("ISAACSIM_ROOT", "/isaac-sim")
        ).resolve()
        self._verify_driver_version = (
            os.environ.get("ISAACSIM_VERIFY_DRIVER_VERSION", "0") != "0"
        )
        self._control_subscription = self.create_subscription(
            String,
            CONTROL_TOPIC,
            self._handle_control,
            10,
        )
        self._ack_publisher = self.create_publisher(String, ACK_TOPIC, 10)
        self._process_monitor = self.create_timer(1.0, self._poll_isaacsim_process)
        self.get_logger().info(f"Isaac Sim manager loaded from {__file__}")
        self.get_logger().info(f"Listening for manager commands on /{CONTROL_TOPIC}")
        self.get_logger().info(f"Publishing manager acknowledgements on /{ACK_TOPIC}")
        self.get_logger().info("Isaac Sim manager service ready")

    def _poll_isaacsim_process(self) -> None:
        if self._isaacsim_process is None:
            return
        return_code = self._isaacsim_process.poll()
        if return_code is None:
            return
        self.get_logger().info(f"Isaac Sim process exited with code {return_code}")
        self._isaacsim_process = None

    def _is_running(self) -> bool:
        return self._isaacsim_process is not None and self._isaacsim_process.poll() is None

    def _build_launch_command(self, mode: str) -> list[str]:
        if mode == START_HEADLESS:
            return ["./runheadless.sh", "-v"]

        launch_command = ["./runapp.sh"]
        if not self._verify_driver_version:
            launch_command.append("--/rtx/verifyDriverVersion/enabled=false")
        return launch_command

    def _start_isaacsim(self, mode: str) -> tuple[bool, str]:
        if self._is_running():
            return False, "Isaac Sim is already running"

        launch_command = self._build_launch_command(mode)
        try:
            self._isaacsim_process = subprocess.Popen(
                launch_command,
                cwd=self._isaacsim_root,
                env=os.environ.copy(),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            self._isaacsim_process = None
            return False, f"Failed to launch Isaac Sim: {exc}"
        except OSError as exc:
            self._isaacsim_process = None
            return False, f"Failed to launch Isaac Sim: {exc}"

        self._last_start_mode = mode
        return True, f"Isaac Sim started in {mode} mode"

    def _stop_isaacsim(self) -> tuple[bool, str]:
        if not self._is_running():
            self._isaacsim_process = None
            return False, "Isaac Sim is not running"

        assert self._isaacsim_process is not None
        try:
            os.killpg(self._isaacsim_process.pid, signal.SIGTERM)
            self._isaacsim_process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(self._isaacsim_process.pid, signal.SIGKILL)
            self._isaacsim_process.wait(timeout=5.0)
        except ProcessLookupError:
            pass

        self._isaacsim_process = None
        return True, "Isaac Sim stopped"

    def _handle_restart(self) -> tuple[bool, str]:
        if self._is_running():
            stopped, stop_message = self._stop_isaacsim()
            if not stopped:
                return False, stop_message

        success, message = self._start_isaacsim(self._last_start_mode)
        if success:
            message = f"Isaac Sim restarted in {self._last_start_mode} mode"
        return success, message

    def _parse_control_message(self, message: String) -> tuple[str, str]:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return uuid.uuid4().hex, message.data.strip().lower()

        request_id = str(payload.get("id") or uuid.uuid4().hex)
        command = str(payload.get("command") or "").strip().lower()
        return request_id, command

    def _publish_ack(self, request_id: str, success: bool, result: str) -> None:
        ack = String()
        ack.data = json.dumps(
            {
                "id": request_id,
                "success": success,
                "message": result,
            },
            separators=(",", ":"),
        )
        self._ack_publisher.publish(ack)
        self.get_logger().info(f"Published manager ack: {ack.data}")

    def _handle_control(self, message: String) -> None:
        request_id, command = self._parse_control_message(message)
        self.get_logger().info(
            f"Received manager command id={request_id!r} command={command!r}"
        )
        if command == "start":
            success, result = self._start_isaacsim(START_HEADED)
        elif command == "start_headless":
            success, result = self._start_isaacsim(START_HEADLESS)
        elif command in {"stop", "kill"}:
            success, result = self._stop_isaacsim()
        elif command == "restart":
            success, result = self._handle_restart()
        else:
            success = False
            result = f"Unsupported manager command: {message.data!r}"
            self.get_logger().warning(result)
            self._publish_ack(request_id, success, result)
            return

        if success:
            self.get_logger().info(result)
        else:
            self.get_logger().warning(result)
        self._publish_ack(request_id, success, result)


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
