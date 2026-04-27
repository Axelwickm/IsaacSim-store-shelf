#!/usr/bin/env python3

import time
import random
import math
from pathlib import Path

import cv2
from geometry_msgs.msg import PointStamped, PoseStamped
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
import tf2_geometry_msgs
import tf2_ros

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
DEFAULT_CAMERA_INFO_TOPIC = "/camera/camera_info"
DEFAULT_SELECTED_ITEM_POINT_TOPIC = "/vision/selected_item_point"
DEFAULT_SELECTED_ITEM_MOVEIT_POINT_TOPIC = "/vision/selected_item_moveit_point"
DEFAULT_SELECTED_ITEM_POSE_TOPIC = "/vision/selected_item_pose"
DEFAULT_TARGET_FRAME_ID = "world"
DEFAULT_MOVEIT_TARGET_FRAME_ID = "yumi_body"
DEFAULT_QUERY_ALPHA_THRESHOLD = 0.2
DEFAULT_QUERY_RENDER_ALPHA_THRESHOLD = 0.2
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
        self.declare_parameter("camera_info_topic", DEFAULT_CAMERA_INFO_TOPIC)
        self.declare_parameter("selected_item_point_topic", DEFAULT_SELECTED_ITEM_POINT_TOPIC)
        self.declare_parameter(
            "selected_item_moveit_point_topic",
            DEFAULT_SELECTED_ITEM_MOVEIT_POINT_TOPIC,
        )
        self.declare_parameter("selected_item_pose_topic", DEFAULT_SELECTED_ITEM_POSE_TOPIC)
        self.declare_parameter("target_frame_id", DEFAULT_TARGET_FRAME_ID)
        self.declare_parameter("moveit_target_frame_id", DEFAULT_MOVEIT_TARGET_FRAME_ID)
        self.declare_parameter("query_alpha_threshold", DEFAULT_QUERY_ALPHA_THRESHOLD)
        self.declare_parameter(
            "query_render_alpha_threshold",
            DEFAULT_QUERY_RENDER_ALPHA_THRESHOLD,
        )
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
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        selected_item_point_topic = str(
            self.get_parameter("selected_item_point_topic").value
        )
        selected_item_moveit_point_topic = str(
            self.get_parameter("selected_item_moveit_point_topic").value
        )
        selected_item_pose_topic = str(
            self.get_parameter("selected_item_pose_topic").value
        )
        self._target_frame_id = str(self.get_parameter("target_frame_id").value)
        self._moveit_target_frame_id = str(
            self.get_parameter("moveit_target_frame_id").value
        )
        self._query_alpha_threshold = float(
            self.get_parameter("query_alpha_threshold").value
        )
        self._query_render_alpha_threshold = float(
            self.get_parameter("query_render_alpha_threshold").value
        )
        self._image_size = int(self.get_parameter("image_size").value)
        self._use_mixed_precision = bool(self.get_parameter("use_mixed_precision").value)
        self._log_every = max(int(self.get_parameter("log_every").value), 1)
        self._camera_info: CameraInfo | None = None

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
        self._selected_item_pose_publisher = self.create_publisher(
            PoseStamped,
            selected_item_pose_topic,
            10,
        )
        self._selected_item_point_publisher = self.create_publisher(
            PointStamped,
            selected_item_point_topic,
            10,
        )
        self._selected_item_moveit_point_publisher = self.create_publisher(
            PointStamped,
            selected_item_moveit_point_topic,
            10,
        )
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._handle_camera_info,
            10,
        )
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
        self.get_logger().info(
            "Publishing selected item point "
            f"on {selected_item_point_topic} in frame {self._target_frame_id}; "
            f"MoveIt point on {selected_item_moveit_point_topic} "
            f"in frame {self._moveit_target_frame_id}; "
            f"legacy pose mirror on {selected_item_pose_topic}"
        )

    def _handle_camera_info(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _publish_selected_item_pose(
        self,
        outputs: dict,
        image_header,
        source_width: int,
        source_height: int,
    ) -> None:
        if self._camera_info is None:
            if self._frame_count == 0 or self._frame_count % self._log_every == 0:
                self.get_logger().warning(
                    "Skipping selected item pose: no CameraInfo received yet"
                )
            return

        patch_alpha = outputs["patch_alpha"][0].detach().float()
        centers = outputs["centers"][0].detach().float()
        sizes = outputs["sizes"][0].detach().float()
        predicted_depth = outputs["predicted_depth"][0].detach().float()
        while predicted_depth.ndim > 2:
            predicted_depth = predicted_depth.squeeze(0)
        patch_alpha_cpu = patch_alpha.cpu()
        alpha_scores = patch_alpha_cpu.mean(dim=(1, 2))
        valid_query_indices = torch.nonzero(
            alpha_scores > self._query_alpha_threshold,
            as_tuple=False,
        ).flatten()
        if valid_query_indices.numel() == 0:
            if self._frame_count == 0 or self._frame_count % self._log_every == 0:
                self.get_logger().warning(
                    "Skipping selected item pose: no query exceeded "
                    f"query_alpha_threshold={self._query_alpha_threshold:.3f}"
                )
            return

        query_index = int(random.choice(valid_query_indices.tolist()))
        query_alpha_score = float(alpha_scores[query_index].item())
        query_alpha = self._render_query_alpha_mask(
            patch_alpha[query_index],
            centers[query_index],
            sizes[query_index],
        )
        mask = query_alpha > self._query_render_alpha_threshold
        if not torch.any(mask):
            if self._frame_count == 0 or self._frame_count % self._log_every == 0:
                self.get_logger().warning(
                    f"Skipping selected item pose: query {query_index} rendered mask is empty"
                )
            return

        ys, xs = torch.nonzero(mask, as_tuple=True)
        weights = query_alpha[ys, xs]
        model_u = float((xs.float() * weights).sum().item() / weights.sum().item())
        model_v = float((ys.float() * weights).sum().item() / weights.sum().item())
        range_m = float(predicted_depth[ys, xs].mean().item())
        if range_m <= 0.0:
            if self._frame_count == 0 or self._frame_count % self._log_every == 0:
                self.get_logger().warning(
                    f"Skipping selected item pose: query {query_index} range={range_m:.3f}"
                )
            return

        scale_x = float(source_width) / float(self._image_size)
        scale_y = float(source_height) / float(self._image_size)
        u = model_u * scale_x
        v = model_v * scale_y

        fx = float(self._camera_info.k[0])
        fy = float(self._camera_info.k[4])
        cx = float(self._camera_info.k[2])
        cy = float(self._camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            if self._frame_count == 0 or self._frame_count % self._log_every == 0:
                self.get_logger().warning("Skipping selected item pose: invalid CameraInfo K")
            return

        x_normalized = (u - cx) / fx
        y_normalized = (v - cy) / fy
        depth_m = range_m / math.sqrt(
            x_normalized * x_normalized + y_normalized * y_normalized + 1.0
        )

        camera_pose = PoseStamped()
        # In fixed-scene store demo mode TF can lag image stamps slightly; latest
        # TF avoids future extrapolation while cart motion is disabled.
        camera_pose.header.stamp = rclpy.time.Time().to_msg()
        camera_pose.header.frame_id = self._camera_info.header.frame_id or image_header.frame_id
        camera_pose.pose.position.x = x_normalized * depth_m
        camera_pose.pose.position.y = y_normalized * depth_m
        camera_pose.pose.position.z = depth_m
        camera_pose.pose.orientation.w = 1.0

        try:
            world_pose = self._tf_buffer.transform(
                camera_pose,
                self._target_frame_id,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception as error:
            if self._frame_count == 0 or self._frame_count % self._log_every == 0:
                self.get_logger().warning(
                    "Could not transform selected item pose "
                    f"from {camera_pose.header.frame_id} to {self._target_frame_id}: {error}"
                )
            return

        try:
            camera_roundtrip_pose = self._tf_buffer.transform(
                world_pose,
                camera_pose.header.frame_id,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            if camera_roundtrip_pose.pose.position.z != 0.0:
                roundtrip_u = (
                    fx
                    * camera_roundtrip_pose.pose.position.x
                    / camera_roundtrip_pose.pose.position.z
                    + cx
                )
                roundtrip_v = (
                    fy
                    * camera_roundtrip_pose.pose.position.y
                    / camera_roundtrip_pose.pose.position.z
                    + cy
                )
            else:
                roundtrip_u = float("nan")
                roundtrip_v = float("nan")
        except Exception:
            camera_roundtrip_pose = None
            roundtrip_u = float("nan")
            roundtrip_v = float("nan")

        camera_origin_world = None
        target_from_camera_world = None
        optical_forward_dot = float("nan")
        try:
            camera_origin_pose = PoseStamped()
            camera_origin_pose.header = camera_pose.header
            camera_origin_pose.pose.orientation.w = 1.0
            camera_origin_world = self._tf_buffer.transform(
                camera_origin_pose,
                self._target_frame_id,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )

            camera_forward_pose = PoseStamped()
            camera_forward_pose.header = camera_pose.header
            camera_forward_pose.pose.position.z = 1.0
            camera_forward_pose.pose.orientation.w = 1.0
            camera_forward_world = self._tf_buffer.transform(
                camera_forward_pose,
                self._target_frame_id,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )

            forward_x = (
                camera_forward_world.pose.position.x
                - camera_origin_world.pose.position.x
            )
            forward_y = (
                camera_forward_world.pose.position.y
                - camera_origin_world.pose.position.y
            )
            forward_z = (
                camera_forward_world.pose.position.z
                - camera_origin_world.pose.position.z
            )
            target_from_camera_world = (
                world_pose.pose.position.x - camera_origin_world.pose.position.x,
                world_pose.pose.position.y - camera_origin_world.pose.position.y,
                world_pose.pose.position.z - camera_origin_world.pose.position.z,
            )
            optical_forward_dot = (
                target_from_camera_world[0] * forward_x
                + target_from_camera_world[1] * forward_y
                + target_from_camera_world[2] * forward_z
            )
        except Exception:
            pass

        world_point = PointStamped()
        world_point.header = world_pose.header
        world_point.point = world_pose.pose.position
        self._selected_item_point_publisher.publish(world_point)
        self._selected_item_pose_publisher.publish(world_pose)

        moveit_point = None
        try:
            moveit_point = PointStamped()
            moveit_point.header = world_point.header
            moveit_point.point = world_point.point
            moveit_point = self._tf_buffer.transform(
                moveit_point,
                self._moveit_target_frame_id,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            self._selected_item_moveit_point_publisher.publish(moveit_point)
        except Exception as error:
            if self._frame_count == 0 or self._frame_count % self._log_every == 0:
                self.get_logger().warning(
                    "Could not transform selected item point "
                    f"from {world_point.header.frame_id} to {self._moveit_target_frame_id}: {error}"
                )

        if self._frame_count == 0 or self._frame_count % self._log_every == 0:
            self.get_logger().info(
                f"Published selected item query={query_index} "
                f"mean_alpha={query_alpha_score:.3f} pixel=({u:.1f},{v:.1f}) "
                f"range={range_m:.3f}m z_depth={depth_m:.3f}m "
                f"camera_xyz=({camera_pose.pose.position.x:.3f}, "
                f"{camera_pose.pose.position.y:.3f}, "
                f"{camera_pose.pose.position.z:.3f}) "
                f"world_xyz=({world_point.point.x:.3f}, "
                f"{world_point.point.y:.3f}, "
                f"{world_point.point.z:.3f}) "
                f"roundtrip_pixel=({roundtrip_u:.1f},{roundtrip_v:.1f}) "
                f"camera_world={self._format_pose_position(camera_origin_world)} "
                f"target_minus_camera_world={self._format_vec3(target_from_camera_world)} "
                f"optical_forward_dot={optical_forward_dot:.3f} "
                f"moveit_xyz={self._format_point(moveit_point)}"
            )

    def _format_pose_position(self, pose: PoseStamped | None) -> str:
        if pose is None:
            return "(nan,nan,nan)"
        return (
            f"({pose.pose.position.x:.3f},"
            f"{pose.pose.position.y:.3f},"
            f"{pose.pose.position.z:.3f})"
        )

    def _format_vec3(self, value: tuple[float, float, float] | None) -> str:
        if value is None:
            return "(nan,nan,nan)"
        return f"({value[0]:.3f},{value[1]:.3f},{value[2]:.3f})"

    def _format_point(self, point: PointStamped | None) -> str:
        if point is None:
            return "(nan,nan,nan)"
        return f"({point.point.x:.3f},{point.point.y:.3f},{point.point.z:.3f})"

    def _render_query_alpha_mask(
        self,
        patch_alpha: torch.Tensor,
        center: torch.Tensor,
        size: torch.Tensor,
    ) -> torch.Tensor:
        device = patch_alpha.device
        ys = torch.linspace(-1.0, 1.0, self._image_size, device=device)
        xs = torch.linspace(-1.0, 1.0, self._image_size, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        local_x = ((grid_x - center[0]) / size[0]).clamp(-1.1, 1.1)
        local_y = ((grid_y - center[1]) / size[1]).clamp(-1.1, 1.1)
        sample_grid = torch.stack([local_x, local_y], dim=-1).view(
            1,
            self._image_size,
            self._image_size,
            2,
        )
        return torch.nn.functional.grid_sample(
            patch_alpha.view(1, 1, patch_alpha.shape[0], patch_alpha.shape[1]),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(self._image_size, self._image_size)

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
        self._publish_selected_item_pose(
            outputs,
            header,
            source_width=message.width,
            source_height=message.height,
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
