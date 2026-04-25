import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image


DEFAULT_DEBUG_IMAGE_TOPIC = "/vision/debug_image"
PANEL_SIZE_POINTS = 320
PANEL_MARGIN_POINTS = 24
MAX_PLACEMENT_ATTEMPTS = 120


def _image_message_to_rgba_array(message: Image) -> np.ndarray:
    image = np.frombuffer(message.data, dtype=np.uint8)
    channel_count = int(message.step) // max(int(message.width), 1)
    image = image.reshape((message.height, message.width, channel_count))

    if message.encoding == "rgb8":
        rgb = image[:, :, :3]
    elif message.encoding == "rgba8":
        rgb = image[:, :, :3]
    elif message.encoding == "bgr8":
        rgb = image[:, :, :3][:, :, ::-1]
    elif message.encoding == "bgra8":
        rgb = image[:, :, :4][:, :, [2, 1, 0]]
    else:
        raise ValueError(f"Unsupported debug image encoding: {message.encoding!r}")

    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))


class IsaacSimVisionPanel:
    def __init__(self, topic_name: str = DEFAULT_DEBUG_IMAGE_TOPIC) -> None:
        import omni.ui as ui

        self._topic_name = topic_name
        self._ui = ui
        self._window = ui.Window(
            "Vision Debug",
            width=PANEL_SIZE_POINTS,
            height=PANEL_SIZE_POINTS,
            position_x=0,
            position_y=0,
        )
        self._status_model = ui.SimpleStringModel("Waiting for vision frames")
        self._provider = ui.ByteImageProvider()
        self._image_widget = None
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._latest_rgba = None
        self._latest_dimensions = None
        self._placement_attempts = 0
        self._placement_applied = False
        self._executor = SingleThreadedExecutor()
        self._node = Node("isaacsim_vision_panel")
        self._executor.add_node(self._node)
        self._subscription = self._node.create_subscription(
            Image,
            topic_name,
            self._handle_image,
            10,
        )

        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Live vision debug image from ROS 2", height=24)
                ui.StringField(model=self._status_model, height=28, read_only=True)
                self._image_widget = ui.ImageWithProvider(
                    self._provider,
                    fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                    width=ui.Fraction(1),
                    height=ui.Fraction(1),
                )

        self._try_place_bottom_right()

    def _main_window_size(self) -> tuple[int, int]:
        workspace = getattr(self._ui, "Workspace", None)
        if workspace is not None:
            get_width = getattr(workspace, "get_main_window_width", None)
            get_height = getattr(workspace, "get_main_window_height", None)
            if callable(get_width) and callable(get_height):
                return int(get_width()), int(get_height())

            get_width = getattr(workspace, "getMainWindowWidth", None)
            get_height = getattr(workspace, "getMainWindowHeight", None)
            if callable(get_width) and callable(get_height):
                return int(get_width()), int(get_height())

        return (
            int(self._ui.get_main_window_width()),
            int(self._ui.get_main_window_height()),
        )

    def _try_place_bottom_right(self) -> None:
        if self._placement_applied:
            return
        if self._placement_attempts >= MAX_PLACEMENT_ATTEMPTS:
            return
        self._placement_attempts += 1

        main_width, main_height = self._main_window_size()
        if main_width <= 0 or main_height <= 0:
            return

        self._window.width = PANEL_SIZE_POINTS
        self._window.height = PANEL_SIZE_POINTS
        self._window.position_x = max(
            main_width - PANEL_SIZE_POINTS - PANEL_MARGIN_POINTS,
            PANEL_MARGIN_POINTS,
        )
        self._window.position_y = max(
            main_height - PANEL_SIZE_POINTS - PANEL_MARGIN_POINTS,
            PANEL_MARGIN_POINTS,
        )
        self._placement_applied = True

    def _handle_image(self, message: Image) -> None:
        try:
            rgba = _image_message_to_rgba_array(message)
        except Exception as error:
            self._status_model.set_value(f"Decode error on {self._topic_name}: {error}")
            return

        self._latest_rgba = rgba
        self._latest_dimensions = [int(message.width), int(message.height)]
        self._last_frame_time = time.monotonic()
        self._frame_count += 1
        self._status_model.set_value(
            f"Frames: {self._frame_count}  Topic: {self._topic_name}  "
            f"Size: {message.width}x{message.height}  Frame: {message.header.frame_id}"
        )

    def update(self) -> None:
        self._executor.spin_once(timeout_sec=0.0)
        self._try_place_bottom_right()
        if self._latest_rgba is None or self._latest_dimensions is None:
            return
        self._provider.set_data_array(self._latest_rgba, self._latest_dimensions)
        self._latest_rgba = None

        frame_age = time.monotonic() - self._last_frame_time
        if frame_age > 1.0:
            self._status_model.set_value(
                f"Stale stream on {self._topic_name}  Last frame age: {frame_age:.1f}s"
            )

    def close(self) -> None:
        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        self._node.destroy_node()
        self._window.visible = False
        self._window = None
