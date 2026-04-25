#!/usr/bin/env python3

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image

from vision.checkpoints import load_checkpoint
from vision.dataset import DEFAULT_IMAGE_SIZE, RGB_MEAN, RGB_STD
from vision.model import HIDDEN_DIM, LATENT_DIM, NUM_QUERIES, PATCH_SIZE, create_query_model, model_device


DEFAULT_CHECKPOINT_DIR = Path("/workspace/checkpoints/vision")
DEFAULT_CAMERA_TOPIC = "/camera/image_raw"
DEFAULT_DEBUG_IMAGE_TOPIC = "/vision/debug_image"
DEFAULT_IDENTITY_TOPIC = "/vision/predicted_identity"
DEFAULT_DEPTH_TOPIC = "/vision/predicted_depth"
DEFAULT_DEPTH_VIZ_TOPIC = "/vision/predicted_depth_viz"
DEFAULT_OCCUPANCY_TOPIC = "/vision/predicted_occupancy"
DEFAULT_LOG_EVERY = 30
DEPTH_DISPLAY_MAX_METERS = 2.0


def _to_display_identity(identity_tensor: torch.Tensor) -> np.ndarray:
    identity = identity_tensor.detach().cpu().numpy().astype(np.uint32)
    red = ((identity * 67) % 251).astype(np.uint8)
    green = ((identity * 29 + 71) % 253).astype(np.uint8)
    blue = ((identity * 53 + 149) % 255).astype(np.uint8)
    background = identity == 0
    red[background] = 0
    green[background] = 0
    blue[background] = 0
    return np.stack([red, green, blue], axis=-1)


def _to_display_depth(depth_tensor: torch.Tensor) -> np.ndarray:
    depth = depth_tensor.detach().cpu().numpy().astype(np.float32)
    while depth.ndim > 2:
        depth = depth.squeeze(0)
    valid_mask = depth > 0.0
    normalized = np.clip(depth / DEPTH_DISPLAY_MAX_METERS, 0.0, 1.0)
    normalized = (normalized * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    colored_bgr[~valid_mask] = 0
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _to_display_occupancy(occupancy_tensor: torch.Tensor) -> np.ndarray:
    occupancy = occupancy_tensor.detach().cpu().numpy().astype(np.float32)
    while occupancy.ndim > 2:
        occupancy = occupancy.squeeze(0)
    occupancy = np.clip(occupancy, 0.0, 1.0)
    grayscale = (occupancy * 255.0).astype(np.uint8)
    return np.repeat(grayscale[:, :, None], 3, axis=2)


def _image_message_to_rgb_array(message: Image) -> np.ndarray:
    image = np.frombuffer(message.data, dtype=np.uint8)
    channel_count = int(message.step) // max(int(message.width), 1)
    image = image.reshape((message.height, message.width, channel_count))

    if message.encoding == "rgb8":
        return image[:, :, :3]
    if message.encoding == "rgba8":
        return image[:, :, :3]
    if message.encoding == "bgr8":
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    if message.encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported image encoding: {message.encoding!r}")


def _rgb_array_to_image_message(
    image: np.ndarray,
    *,
    header,
    encoding: str,
) -> Image:
    message = Image()
    message.header = header
    message.height = int(image.shape[0])
    message.width = int(image.shape[1])
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = int(image.strides[0])
    message.data = image.tobytes()
    return message


def _compose_debug_panel(
    rgb: np.ndarray,
    identity_rgb: np.ndarray,
    depth_viz_rgb: np.ndarray,
    occupancy_rgb: np.ndarray,
) -> np.ndarray:
    rgb_panel = cv2.resize(rgb, (identity_rgb.shape[1], identity_rgb.shape[0]))
    top_row = np.hstack([rgb_panel, identity_rgb])
    bottom_row = np.hstack([depth_viz_rgb, occupancy_rgb])
    return np.ascontiguousarray(np.vstack([top_row, bottom_row]))


class VisionInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_inference")
        self.declare_parameter("checkpoint_dir", str(DEFAULT_CHECKPOINT_DIR))
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("image_topic", DEFAULT_CAMERA_TOPIC)
        self.declare_parameter("debug_image_topic", DEFAULT_DEBUG_IMAGE_TOPIC)
        self.declare_parameter("identity_topic", DEFAULT_IDENTITY_TOPIC)
        self.declare_parameter("depth_topic", DEFAULT_DEPTH_TOPIC)
        self.declare_parameter("depth_viz_topic", DEFAULT_DEPTH_VIZ_TOPIC)
        self.declare_parameter("occupancy_topic", DEFAULT_OCCUPANCY_TOPIC)
        self.declare_parameter("image_size", DEFAULT_IMAGE_SIZE)
        self.declare_parameter("use_mixed_precision", True)
        self.declare_parameter("log_every", DEFAULT_LOG_EVERY)

        checkpoint_dir = Path(str(self.get_parameter("checkpoint_dir").value)).resolve()
        checkpoint_path_value = str(self.get_parameter("checkpoint_path").value).strip()
        image_topic = str(self.get_parameter("image_topic").value)
        debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        identity_topic = str(self.get_parameter("identity_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        depth_viz_topic = str(self.get_parameter("depth_viz_topic").value)
        occupancy_topic = str(self.get_parameter("occupancy_topic").value)
        self._image_size = int(self.get_parameter("image_size").value)
        self._use_mixed_precision = bool(self.get_parameter("use_mixed_precision").value)
        self._log_every = max(int(self.get_parameter("log_every").value), 1)

        latest_checkpoint_path = checkpoint_dir / "latest.pt"
        checkpoint_path = (
            Path(checkpoint_path_value).resolve()
            if checkpoint_path_value
            else latest_checkpoint_path
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._device = model_device()
        self._model = create_query_model().to(self._device)
        checkpoint_config = {
            "num_queries": NUM_QUERIES,
            "latent_dim": LATENT_DIM,
            "patch_size": PATCH_SIZE,
            "hidden_dim": HIDDEN_DIM,
        }
        checkpoint_metadata = load_checkpoint(
            checkpoint_path,
            self._model,
            expected_config=checkpoint_config,
            map_location=self._device,
        )
        self._model.eval()

        self._debug_image_publisher = self.create_publisher(Image, debug_image_topic, 10)
        self._identity_publisher = self.create_publisher(Image, identity_topic, 10)
        self._depth_publisher = self.create_publisher(Image, depth_topic, 10)
        self._depth_viz_publisher = self.create_publisher(Image, depth_viz_topic, 10)
        self._occupancy_publisher = self.create_publisher(Image, occupancy_topic, 10)
        self._image_subscription = self.create_subscription(
            Image,
            image_topic,
            self._handle_image,
            10,
        )
        self._frame_count = 0
        self._inference_time_sum = 0.0

        self.get_logger().info(
            f"Loaded checkpoint from {checkpoint_path} at "
            f"epoch={checkpoint_metadata['epoch']} "
            f"global_step={checkpoint_metadata['global_step']}"
        )
        self.get_logger().info(
            "Vision inference ready "
            f"(image_topic={image_topic}, image_size={self._image_size}, device={self._device})"
        )
        self.get_logger().info(
            "Publishing debug outputs on "
            f"{debug_image_topic}, {identity_topic}, {depth_topic}, "
            f"{depth_viz_topic}, and {occupancy_topic}"
        )

    def _handle_image(self, message: Image) -> None:
        try:
            rgb = _image_message_to_rgb_array(message)
        except Exception as error:
            self.get_logger().error(f"Failed to decode image frame: {error}")
            return

        resized_rgb = cv2.resize(
            rgb,
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_AREA,
        )
        rgb_tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float() / 255.0
        normalized_rgb = (rgb_tensor - RGB_MEAN) / RGB_STD
        pixel_values = normalized_rgb.unsqueeze(0).to(self._device)

        start_time = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(
                device_type=self._device.type,
                enabled=self._use_mixed_precision and self._device.type == "cuda",
            ):
                outputs = self._model(pixel_values)
        inference_elapsed = time.perf_counter() - start_time

        identity_rgb = _to_display_identity(outputs["predicted_identity"][0])
        predicted_depth = outputs["predicted_depth"][0].detach().cpu().numpy().astype(
            np.float32
        )
        while predicted_depth.ndim > 2:
            predicted_depth = predicted_depth.squeeze(0)
        depth_viz_rgb = _to_display_depth(outputs["predicted_depth"][0])
        occupancy_rgb = _to_display_occupancy(outputs["predicted_occupancy"][0])
        debug_image = _compose_debug_panel(
            resized_rgb,
            identity_rgb,
            depth_viz_rgb,
            occupancy_rgb,
        )

        header = message.header
        self._debug_image_publisher.publish(
            _rgb_array_to_image_message(debug_image, header=header, encoding="rgb8")
        )
        self._identity_publisher.publish(
            _rgb_array_to_image_message(identity_rgb, header=header, encoding="rgb8")
        )
        self._depth_publisher.publish(
            _rgb_array_to_image_message(
                np.ascontiguousarray(predicted_depth),
                header=header,
                encoding="32FC1",
            )
        )
        self._depth_viz_publisher.publish(
            _rgb_array_to_image_message(depth_viz_rgb, header=header, encoding="rgb8")
        )
        self._occupancy_publisher.publish(
            _rgb_array_to_image_message(occupancy_rgb, header=header, encoding="rgb8")
        )

        self._frame_count += 1
        self._inference_time_sum += inference_elapsed
        if self._frame_count == 1 or self._frame_count % self._log_every == 0:
            average_inference_ms = 1000.0 * self._inference_time_sum / self._frame_count
            self.get_logger().info(
                f"Processed {self._frame_count} frames "
                f"(last_inference_ms={1000.0 * inference_elapsed:.1f}, "
                f"avg_inference_ms={average_inference_ms:.1f}, "
                f"identity_nonzero={(outputs['predicted_identity'][0] > 0).sum().item()}, "
                f"depth_max={outputs['predicted_depth'][0].max().item():.3f}, "
                f"occupancy_max={outputs['predicted_occupancy'][0].max().item():.3f})"
            )


def main() -> None:
    rclpy.init()
    node = VisionInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
