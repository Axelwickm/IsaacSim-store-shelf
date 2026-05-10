#!/usr/bin/env python3

import json
import math
import os
import random
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from geometry_msgs.msg import PointStamped, PoseStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import torch
import tf2_geometry_msgs
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

from vision.checkpoints import load_checkpoint
from vision.dataset import DEFAULT_IMAGE_SIZE, RGB_MEAN, RGB_STD
from vision.model import (
    HIDDEN_DIM,
    LATENT_DIM,
    NUM_QUERIES,
    PATCH_SIZE,
    create_query_model,
    model_device,
    render_billboards,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    _SUMMARY_WRITER_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None
    _SUMMARY_WRITER_IMPORT_ERROR = error


DEFAULT_CHECKPOINT_DIR = Path("/workspace/checkpoints/vision")
DEFAULT_TENSORBOARD_DIR = Path("/workspace/tensorboard/vision")
DEFAULT_REPLAY_DIR = Path("/workspace/replay/vision")
DEFAULT_CAMERA_TOPIC = "/camera/image_raw"
DEFAULT_DEBUG_IMAGE_TOPIC = "/vision/debug_image"
DEFAULT_CAMERA_INFO_TOPIC = "/camera/camera_info"
DEFAULT_SELECTED_CANDIDATE_TOPIC = "/vision/selected_candidate"
DEFAULT_SUGGESTED_ITEM_MARKERS_TOPIC = "/vision/suggested_item_markers"
DEFAULT_GROUND_TRUTH_ITEMS_TOPIC = "/vision/ground_truth_items"
DEFAULT_ARM_STATE_TOPIC = "/motion/arm_state"
DEFAULT_RESET_TOPIC = "/motion/reset"
DEFAULT_TARGET_FRAME_ID = "world"
DEFAULT_MOVEIT_TARGET_FRAME_ID = "yumi_body"
DEFAULT_CAMERA_FRAME_CONVENTION = "ros_optical"
DEFAULT_QUERY_PRESENCE_THRESHOLD = 0.2
DEFAULT_EXPLORATION_EPSILON = 0.10
DEFAULT_LOG_EVERY = 30
DEFAULT_REPLAY_BUFFER_CAPACITY = 8000
DEFAULT_MIN_REPLAY_SIZE = 2
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_TRAIN_STEPS_PER_TICK = 4
DEFAULT_TRAIN_TICK_PERIOD = 0.01
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_GEOMETRY_LOSS_WEIGHT = 1.0
DEFAULT_PRESENCE_LOSS_WEIGHT = 0.2
DEFAULT_DEPTH_LOSS_WEIGHT = 0.2
DEFAULT_CHECKPOINT_SAVE_INTERVAL = 100
DEFAULT_PENDING_CANDIDATE_TIMEOUT = 30.0
DEFAULT_MAX_INFLIGHT_CANDIDATES = 1
DEFAULT_MAX_SUGGESTED_MARKERS = 32
DEFAULT_CUDA_MEMORY_LOG_PERIOD = 30.0


def _default_run_name() -> str:
    return "vision-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def _encode_payload(payload: dict[str, Any]) -> String:
    message = String()
    message.data = json.dumps(payload, sort_keys=True)
    return message


def _decode_payload(message: String) -> dict[str, Any] | None:
    try:
        payload = json.loads(message.data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _parameter_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _value_to_rgb(value: float) -> tuple[float, float, float]:
    value = _clamp01(value)
    return 1.0 - value, value, 0.0


def _arm_side_to_index(arm_side: str) -> int | None:
    normalized = arm_side.strip().lower()
    if normalized == "left":
        return 0
    if normalized == "right":
        return 1
    return None


def _bytes_to_mib(byte_count: int | float) -> float:
    return float(byte_count) / (1024.0 * 1024.0)


def _to_display_depth(depth: torch.Tensor, occupancy: torch.Tensor | None = None) -> np.ndarray:
    depth_np = depth.detach().float().cpu().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    mask_np = None
    if occupancy is not None:
        mask_np = occupancy.detach().float().cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        mask_np = mask_np > 0.05
    valid = depth_np > 1e-6
    if mask_np is not None:
        valid = np.logical_and(valid, mask_np)
    if not np.any(valid):
        return np.zeros((*depth_np.shape, 3), dtype=np.uint8)
    valid_values = depth_np[valid]
    min_depth = float(np.percentile(valid_values, 5.0))
    max_depth = float(np.percentile(valid_values, 95.0))
    if max_depth <= min_depth:
        max_depth = min_depth + 1e-3
    normalized = np.clip((depth_np - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    depth_u8 = (normalized * 255.0).astype(np.uint8)
    display = np.repeat(depth_u8[..., None], 3, axis=2)
    if mask_np is not None:
        display[~mask_np] = 0
    return display


def _to_display_occupancy(occupancy: torch.Tensor) -> np.ndarray:
    occupancy_np = occupancy.detach().float().cpu().numpy()
    if occupancy_np.ndim == 3:
        occupancy_np = occupancy_np[0]
    occupancy_np = np.clip(occupancy_np, 0.0, 1.0)
    return np.repeat((occupancy_np[..., None] * 255.0).astype(np.uint8), 3, axis=2)


def _to_display_identity(identity: torch.Tensor, occupancy: torch.Tensor | None = None) -> np.ndarray:
    identity_np = identity.detach().float().cpu().numpy()
    if identity_np.ndim == 3:
        identity_np = identity_np[0]
    hue = np.mod(identity_np * 137.508, 360.0) / 2.0
    saturation = np.full_like(hue, 180.0)
    value = np.full_like(hue, 255.0)
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.uint8)
    display = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    display[identity_np <= 0.0] = 0
    if occupancy is not None:
        occupancy_np = occupancy.detach().float().cpu().numpy()
        if occupancy_np.ndim == 3:
            occupancy_np = occupancy_np[0]
        display[occupancy_np <= 0.05] = 0
    return display


def _to_display_billboard_values(
    identity: torch.Tensor,
    occupancy: torch.Tensor,
    value_probs: torch.Tensor,
) -> np.ndarray:
    identity_np = identity.detach().float().cpu().numpy()
    if identity_np.ndim == 3:
        identity_np = identity_np[0]
    occupancy_np = occupancy.detach().float().cpu().numpy()
    if occupancy_np.ndim == 3:
        occupancy_np = occupancy_np[0]
    values_np = value_probs.detach().float().cpu().numpy()
    values_np = np.clip(values_np, 0.0, 1.0)

    display = np.zeros((*identity_np.shape, 3), dtype=np.uint8)
    for query_index, query_values in enumerate(values_np.tolist()):
        mask = (np.rint(identity_np).astype(np.int32) == query_index + 1) & (
            occupancy_np > 0.05
        )
        if not np.any(mask):
            continue
        if isinstance(query_values, list) and len(query_values) >= 2:
            green = _clamp01(float(query_values[0]))
            red = _clamp01(float(query_values[1]))
            blue = 0.0
        else:
            red, green, blue = _value_to_rgb(float(query_values))
        display[mask] = (
            int(round(red * 255.0)),
            int(round(green * 255.0)),
            int(round(blue * 255.0)),
        )
    return display


def _compose_debug_panel(
    rgb: np.ndarray,
    identity: torch.Tensor,
    depth: torch.Tensor,
    occupancy: torch.Tensor,
    value_probs: torch.Tensor,
    selected_mask: np.ndarray | None,
) -> np.ndarray:
    identity_display = _to_display_identity(identity, occupancy)
    value_display = _to_display_billboard_values(identity, occupancy, value_probs)
    occupancy_display = _to_display_occupancy(occupancy)
    rgb_display = np.ascontiguousarray(rgb.copy())
    if selected_mask is not None:
        mask = cv2.resize(
            selected_mask.astype(np.float32),
            (rgb_display.shape[1], rgb_display.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        tint = np.zeros_like(rgb_display)
        tint[:, :, 1] = 255
        alpha = np.clip(mask[..., None] * 0.55, 0.0, 0.55)
        rgb_display = (rgb_display.astype(np.float32) * (1.0 - alpha) + tint * alpha).astype(
            np.uint8
        )
    top = np.concatenate([rgb_display, identity_display], axis=1)
    bottom = np.concatenate([value_display, occupancy_display], axis=1)
    return np.concatenate([top, bottom], axis=0)


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


def _rgb_array_to_image_message(image: np.ndarray, *, header, encoding: str) -> Image:
    message = Image()
    message.header = header
    message.height = int(image.shape[0])
    message.width = int(image.shape[1])
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = int(image.strides[0])
    message.data = image.tobytes()
    return message
class VisionInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_inference")
        self.declare_parameter("checkpoint_dir", str(DEFAULT_CHECKPOINT_DIR))
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("image_topic", DEFAULT_CAMERA_TOPIC)
        self.declare_parameter("debug_image_topic", DEFAULT_DEBUG_IMAGE_TOPIC)
        self.declare_parameter("camera_info_topic", DEFAULT_CAMERA_INFO_TOPIC)
        self.declare_parameter("selected_candidate_topic", DEFAULT_SELECTED_CANDIDATE_TOPIC)
        self.declare_parameter("suggested_item_markers_topic", DEFAULT_SUGGESTED_ITEM_MARKERS_TOPIC)
        self.declare_parameter("ground_truth_items_topic", DEFAULT_GROUND_TRUTH_ITEMS_TOPIC)
        self.declare_parameter("arm_state_topic", DEFAULT_ARM_STATE_TOPIC)
        self.declare_parameter("target_frame_id", DEFAULT_TARGET_FRAME_ID)
        self.declare_parameter("moveit_target_frame_id", DEFAULT_MOVEIT_TARGET_FRAME_ID)
        self.declare_parameter("camera_frame_convention", DEFAULT_CAMERA_FRAME_CONVENTION)
        self.declare_parameter("query_presence_threshold", DEFAULT_QUERY_PRESENCE_THRESHOLD)
        self.declare_parameter("exploration_epsilon", DEFAULT_EXPLORATION_EPSILON)
        self.declare_parameter("image_size", DEFAULT_IMAGE_SIZE)
        self.declare_parameter("use_mixed_precision", True)
        self.declare_parameter("log_every", DEFAULT_LOG_EVERY)
        self.declare_parameter("online_training_enabled", False)
        self.declare_parameter("replay_buffer_capacity", DEFAULT_REPLAY_BUFFER_CAPACITY)
        self.declare_parameter("min_replay_size", DEFAULT_MIN_REPLAY_SIZE)
        self.declare_parameter("train_batch_size", DEFAULT_TRAIN_BATCH_SIZE)
        self.declare_parameter("online_learning_rate", DEFAULT_LEARNING_RATE)
        self.declare_parameter("geometry_loss_weight", DEFAULT_GEOMETRY_LOSS_WEIGHT)
        self.declare_parameter("presence_loss_weight", DEFAULT_PRESENCE_LOSS_WEIGHT)
        self.declare_parameter("depth_loss_weight", DEFAULT_DEPTH_LOSS_WEIGHT)
        self.declare_parameter("freeze_backbone", True)
        self.declare_parameter("tensorboard_log_dir", str(DEFAULT_TENSORBOARD_DIR))
        self.declare_parameter("tensorboard_run_name", "")
        self.declare_parameter("replay_dir", str(DEFAULT_REPLAY_DIR))
        self.declare_parameter("checkpoint_reload_period_sec", 5.0)
        self.declare_parameter("checkpoint_save_interval", DEFAULT_CHECKPOINT_SAVE_INTERVAL)
        self.declare_parameter("pending_candidate_timeout_sec", DEFAULT_PENDING_CANDIDATE_TIMEOUT)
        self.declare_parameter("max_inflight_candidates", DEFAULT_MAX_INFLIGHT_CANDIDATES)
        self.declare_parameter("max_suggested_markers", DEFAULT_MAX_SUGGESTED_MARKERS)
        self.declare_parameter("cuda_memory_log_period_sec", DEFAULT_CUDA_MEMORY_LOG_PERIOD)

        checkpoint_dir = Path(str(self.get_parameter("checkpoint_dir").value)).resolve()
        checkpoint_path_value = str(self.get_parameter("checkpoint_path").value).strip()
        image_topic = str(self.get_parameter("image_topic").value)
        debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        selected_candidate_topic = str(self.get_parameter("selected_candidate_topic").value)
        suggested_item_markers_topic = str(
            self.get_parameter("suggested_item_markers_topic").value
        )
        ground_truth_items_topic = str(self.get_parameter("ground_truth_items_topic").value)
        arm_state_topic = str(self.get_parameter("arm_state_topic").value)
        self._target_frame_id = str(self.get_parameter("target_frame_id").value)
        self._moveit_target_frame_id = str(self.get_parameter("moveit_target_frame_id").value)
        self._camera_frame_convention = str(
            self.get_parameter("camera_frame_convention").value
        ).strip()
        if self._camera_frame_convention not in {"usd_camera", "ros_optical"}:
            raise ValueError(
                "camera_frame_convention must be 'usd_camera' or 'ros_optical', "
                f"got {self._camera_frame_convention!r}"
            )
        self._query_presence_threshold = float(
            self.get_parameter("query_presence_threshold").value
        )
        self._exploration_epsilon = float(self.get_parameter("exploration_epsilon").value)
        self._image_size = int(self.get_parameter("image_size").value)
        self._use_mixed_precision = _parameter_bool(
            self.get_parameter("use_mixed_precision").value
        )
        self._log_every = max(int(self.get_parameter("log_every").value), 1)
        self._online_training_enabled = _parameter_bool(
            self.get_parameter("online_training_enabled").value
        )
        self._replay_buffer_capacity = max(
            int(self.get_parameter("replay_buffer_capacity").value),
            1,
        )
        self._min_replay_size = max(int(self.get_parameter("min_replay_size").value), 1)
        self._train_batch_size = max(int(self.get_parameter("train_batch_size").value), 1)
        self._online_learning_rate = float(
            self.get_parameter("online_learning_rate").value
        )
        self._geometry_loss_weight = float(
            self.get_parameter("geometry_loss_weight").value
        )
        self._presence_loss_weight = float(
            self.get_parameter("presence_loss_weight").value
        )
        self._depth_loss_weight = float(self.get_parameter("depth_loss_weight").value)
        self._freeze_backbone = _parameter_bool(self.get_parameter("freeze_backbone").value)
        self._tensorboard_log_dir = Path(
            str(self.get_parameter("tensorboard_log_dir").value)
        ).resolve()
        self._tensorboard_run_name = str(
            self.get_parameter("tensorboard_run_name").value
        ).strip()
        self._replay_dir = Path(str(self.get_parameter("replay_dir").value)).resolve()
        self._checkpoint_reload_period_sec = max(
            float(self.get_parameter("checkpoint_reload_period_sec").value),
            0.5,
        )
        self._checkpoint_save_interval = max(
            int(self.get_parameter("checkpoint_save_interval").value),
            1,
        )
        self._pending_candidate_timeout_sec = max(
            float(self.get_parameter("pending_candidate_timeout_sec").value),
            1.0,
        )
        self._max_inflight_candidates = max(
            int(self.get_parameter("max_inflight_candidates").value),
            1,
        )
        self._max_suggested_markers = max(
            int(self.get_parameter("max_suggested_markers").value),
            1,
        )
        self._cuda_memory_log_period_sec = max(
            float(self.get_parameter("cuda_memory_log_period_sec").value),
            0.0,
        )
        self._camera_info: CameraInfo | None = None

        latest_checkpoint_path = checkpoint_dir / "latest.pt"
        checkpoint_path = (
            Path(checkpoint_path_value).resolve()
            if checkpoint_path_value
            else latest_checkpoint_path
        )

        self._checkpoint_dir = checkpoint_dir
        self._latest_checkpoint_path = latest_checkpoint_path
        self._device = model_device()
        self._model = create_query_model().to(self._device)
        self._checkpoint_config = {
            "num_queries": NUM_QUERIES,
            "patch_size": PATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "value_in_slot_head": True,
            "billboard_rendering": "center_sampled_slots_size2p5_v6",
        }

        self._optimizer: torch.optim.Optimizer | None = None
        self._scaler = None

        checkpoint_metadata = {
            "epoch": 0,
            "global_step": 0,
            "config": dict(self._checkpoint_config),
            "missing_keys": [],
            "unexpected_keys": [],
        }
        checkpoint_loaded = False
        if checkpoint_path.exists():
            try:
                checkpoint_metadata = load_checkpoint(
                    checkpoint_path,
                    self._model,
                    expected_config=self._checkpoint_config,
                    map_location=self._device,
                    strict=False,
                )
                checkpoint_loaded = True
            except Exception as error:
                self.get_logger().warning(
                    f"Skipping incompatible vision checkpoint {checkpoint_path}: {error}"
                )
        self._model.eval()
        self._train_step = int(checkpoint_metadata["global_step"])
        self._writer = None
        checkpoint_run_name = str(
            (checkpoint_metadata.get("metadata") or {}).get("tensorboard_run_name") or ""
        ).strip()
        if not self._tensorboard_run_name:
            self._tensorboard_run_name = checkpoint_run_name or _default_run_name()
        self._tensorboard_run_dir = self._tensorboard_log_dir / self._tensorboard_run_name
        self._last_checkpoint_mtime_ns = (
            checkpoint_path.stat().st_mtime_ns if checkpoint_path.exists() else 0
        )

        self._debug_image_publisher = self.create_publisher(Image, debug_image_topic, 10)
        self._selected_candidate_publisher = self.create_publisher(
            String,
            selected_candidate_topic,
            10,
        )
        self._suggested_item_markers_publisher = self.create_publisher(
            MarkerArray,
            suggested_item_markers_topic,
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
        self._arm_state_subscription = self.create_subscription(
            String,
            arm_state_topic,
            self._handle_arm_state,
            10,
        )
        self._ground_truth_items_subscription = self.create_subscription(
            String,
            ground_truth_items_topic,
            self._handle_ground_truth_items,
            10,
        )

        self._frame_count = 0
        self._inference_time_sum = 0.0
        self._selection_serial = 0
        self._replay_buffer: deque[dict[str, Any]] = deque(maxlen=self._replay_buffer_capacity)
        self._pending_candidates: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._seen_plan_outcomes: set[tuple[str, str, str]] = set()
        self._ground_truth_items: list[dict[str, Any]] = []
        self._reset_subscription = self.create_subscription(
            String,
            DEFAULT_RESET_TOPIC,
            self._handle_reset,
            10,
        )
        self._outcome_count = 0
        self._last_training_mode_log_monotonic = 0.0
        self._last_replay_wait_log_size = 0
        self._checkpoint_reload_timer = self.create_timer(
            self._checkpoint_reload_period_sec,
            self._maybe_reload_checkpoint,
        )
        self._cuda_memory_timer = None
        if self._device.type == "cuda" and self._cuda_memory_log_period_sec > 0.0:
            self._cuda_memory_timer = self.create_timer(
                self._cuda_memory_log_period_sec,
                self._log_cuda_memory,
            )

        if checkpoint_loaded:
            self.get_logger().info(
                f"Loaded checkpoint from {checkpoint_path} at "
                f"epoch={checkpoint_metadata['epoch']} global_step={checkpoint_metadata['global_step']}"
            )
            self.get_logger().info(
                "Checkpoint restore: "
                f"optimizer_loaded={bool(checkpoint_metadata.get('optimizer_loaded'))}; "
                f"scaler_loaded={bool(checkpoint_metadata.get('scaler_loaded'))}; "
                f"has_optimizer_state={bool(checkpoint_metadata.get('has_optimizer_state'))}; "
                f"has_scaler_state={bool(checkpoint_metadata.get('has_scaler_state'))}"
            )
            saved_metadata = checkpoint_metadata.get("metadata") or {}
            saved_tensorboard_dir = saved_metadata.get("tensorboard_log_dir")
            if saved_tensorboard_dir:
                self.get_logger().info(
                    "Checkpoint metadata: "
                    f"tensorboard_run_name={saved_metadata.get('tensorboard_run_name')}; "
                    f"tensorboard_log_dir={saved_tensorboard_dir}; "
                    f"optimizer={saved_metadata.get('optimizer')}; "
                    f"learning_rate={saved_metadata.get('learning_rate')}; "
                    f"train_batch_size={saved_metadata.get('train_batch_size')}; "
                    f"min_replay_size={saved_metadata.get('min_replay_size')}"
                )
            missing_keys = checkpoint_metadata.get("missing_keys") or []
            unexpected_keys = checkpoint_metadata.get("unexpected_keys") or []
            if missing_keys:
                self.get_logger().info(
                    "Checkpoint missing keys on load: " + ", ".join(sorted(missing_keys))
                )
            if unexpected_keys:
                self.get_logger().info(
                    "Checkpoint unexpected keys on load: "
                    + ", ".join(sorted(unexpected_keys))
                )
        elif not checkpoint_path.exists():
            self.get_logger().warning(
                f"Checkpoint not found at {checkpoint_path}. Starting vision inference from random weights."
            )
        self.get_logger().info(
            "Vision inference ready "
            f"(pid={os.getpid()}, image_topic={image_topic}, image_size={self._image_size}, "
            f"device={self._device})"
        )
        self.get_logger().info(
            "Vision mode: "
            f"{'online-billboard-training' if self._online_training_enabled else 'billboard-inference-only'}; "
            f"candidate topic={selected_candidate_topic}; "
            f"suggested markers topic={suggested_item_markers_topic}; "
            f"arm state topic={arm_state_topic}; "
            f"reset topic={DEFAULT_RESET_TOPIC}"
        )
        self.get_logger().info(
            "Vision mode details: "
            f"exploration_epsilon="
            f"{self._exploration_epsilon if self._online_training_enabled else 0.0:.3f}; "
            f"train_batch_size={self._train_batch_size}; "
            f"geometry_loss_weight={self._geometry_loss_weight:.3f}; "
            f"presence_loss_weight={self._presence_loss_weight:.3f}; "
            f"depth_loss_weight={self._depth_loss_weight:.3f}; "
            f"checkpoint_reload_period_sec={self._checkpoint_reload_period_sec:.1f}; "
            f"camera_frame_convention={self._camera_frame_convention}"
        )
        self._log_cuda_memory()

    def _log_cuda_memory(self) -> None:
        if self._device.type != "cuda":
            return
        device = self._device
        self.get_logger().info(
            "Vision inference CUDA memory: "
            f"pid={os.getpid()}; "
            f"allocated={_bytes_to_mib(torch.cuda.memory_allocated(device)):.1f}MiB; "
            f"reserved={_bytes_to_mib(torch.cuda.memory_reserved(device)):.1f}MiB; "
            f"max_allocated={_bytes_to_mib(torch.cuda.max_memory_allocated(device)):.1f}MiB"
        )

    def _maybe_reload_checkpoint(self) -> None:
        path = self._latest_checkpoint_path
        if not path.exists():
            return
        mtime_ns = path.stat().st_mtime_ns
        if mtime_ns <= self._last_checkpoint_mtime_ns:
            return
        try:
            metadata = load_checkpoint(
                path,
                self._model,
                expected_config=self._checkpoint_config,
                map_location=self._device,
                strict=False,
            )
        except Exception as error:
            self.get_logger().warning(f"Failed to hot-reload vision checkpoint {path}: {error}")
            return
        self._model.eval()
        self._last_checkpoint_mtime_ns = mtime_ns
        self._train_step = int(metadata["global_step"])
        self.get_logger().info(
            f"Hot-reloaded vision checkpoint {path}; train_step={self._train_step}"
        )

    def _handle_camera_info(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _handle_ground_truth_items(self, message: String) -> None:
        payload = _decode_payload(message)
        if payload is None:
            return
        frame_id = str(payload.get("frame_id", "")).strip() or "world"
        items = payload.get("items")
        if not isinstance(items, list):
            return
        ground_truth_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            center_xyz = item.get("center_xyz")
            if not isinstance(center_xyz, list) or len(center_xyz) != 3:
                continue
            ground_truth_items.append(
                {
                    "name": str(item.get("name", "")),
                    "path": str(item.get("path", "")),
                    "point": {
                        "frame_id": frame_id,
                        "xyz": [
                            float(center_xyz[0]),
                            float(center_xyz[1]),
                            float(center_xyz[2]),
                        ],
                    },
                }
            )
        self._ground_truth_items = ground_truth_items

    def _checkpoint_metadata(self) -> dict[str, Any]:
        optimizer_groups = []
        if self._optimizer is not None:
            for group in self._optimizer.param_groups:
                optimizer_groups.append(
                    {
                        "lr": float(group.get("lr", 0.0)),
                        "weight_decay": float(group.get("weight_decay", 0.0)),
                        "betas": list(group.get("betas", ())),
                        "eps": float(group.get("eps", 0.0)),
                    }
                )
        return {
            "tensorboard_base_log_dir": str(self._tensorboard_log_dir),
            "tensorboard_run_name": self._tensorboard_run_name,
            "tensorboard_log_dir": str(self._tensorboard_run_dir),
            "online_training_enabled": self._online_training_enabled,
            "optimizer": type(self._optimizer).__name__ if self._optimizer is not None else None,
            "optimizer_param_groups": optimizer_groups,
            "scaler_enabled": bool(
                self._scaler is not None and getattr(self._scaler, "is_enabled", lambda: False)()
            ),
            "freeze_backbone": self._freeze_backbone,
            "learning_rate": self._online_learning_rate,
            "replay_buffer_capacity": self._replay_buffer_capacity,
            "min_replay_size": self._min_replay_size,
            "train_batch_size": self._train_batch_size,
            "geometry_loss_weight": self._geometry_loss_weight,
            "presence_loss_weight": self._presence_loss_weight,
            "depth_loss_weight": self._depth_loss_weight,
            "checkpoint_save_interval": self._checkpoint_save_interval,
            "exploration_epsilon": self._exploration_epsilon,
        }

    def _write_replay_sample(self, replay_item: dict[str, Any], candidate_id: str, request_id: str) -> None:
        self._replay_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in f"{int(time.time_ns())}-{candidate_id}-{request_id}"
        )
        final_path = self._replay_dir / f"{safe_id}.npz"
        tmp_path = self._replay_dir / f"{safe_id}.tmp.npz"
        metadata = {key: value for key, value in replay_item.items() if key != "rgb"}
        np.savez_compressed(
            tmp_path,
            rgb=replay_item["rgb"],
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        tmp_path.replace(final_path)

    def _normalize_rgb_batch(self, rgb_batch: np.ndarray) -> torch.Tensor:
        pixel_values = torch.from_numpy(rgb_batch).permute(0, 3, 1, 2).float() / 255.0
        normalized = (pixel_values - RGB_MEAN.view(1, 3, 1, 1)) / RGB_STD.view(1, 3, 1, 1)
        return normalized.to(self._device)

    def _serialize_point(self, point: PointStamped) -> dict[str, Any]:
        stamp_ns = int(point.header.stamp.sec) * 1_000_000_000 + int(
            point.header.stamp.nanosec
        )
        return {
            "frame_id": point.header.frame_id,
            "stamp_ns": stamp_ns,
            "xyz": [float(point.point.x), float(point.point.y), float(point.point.z)],
        }

    def _deserialize_point(self, payload: dict[str, Any]) -> PointStamped | None:
        xyz = payload.get("xyz")
        frame_id = str(payload.get("frame_id", "")).strip()
        if not frame_id or not isinstance(xyz, list) or len(xyz) != 3:
            return None
        point = PointStamped()
        point.header.frame_id = frame_id
        point.header.stamp = rclpy.time.Time().to_msg()
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])
        return point

    def _camera_point_from_pixel(
        self,
        x_normalized: float,
        y_normalized: float,
        forward_m: float,
    ) -> tuple[float, float, float, float]:
        if self._camera_frame_convention == "ros_optical":
            return (
                x_normalized * forward_m,
                y_normalized * forward_m,
                forward_m,
                forward_m,
            )
        return (
            x_normalized * forward_m,
            -y_normalized * forward_m,
            -forward_m,
            forward_m,
        )

    def _normalized_pixel_from_camera_point(
        self,
        point: PointStamped,
    ) -> tuple[float, float] | None:
        if self._camera_frame_convention == "ros_optical":
            forward_m = float(point.point.z)
            if forward_m <= 0.0:
                return None
            return float(point.point.x) / forward_m, float(point.point.y) / forward_m
        forward_m = -float(point.point.z)
        if forward_m <= 0.0:
            return None
        return float(point.point.x) / forward_m, -float(point.point.y) / forward_m

    def _camera_forward_from_point(self, point: PointStamped) -> float:
        if self._camera_frame_convention == "ros_optical":
            return float(point.point.z)
        return -float(point.point.z)

    def _project_world_point_to_model_measurement(
        self,
        point_payload: dict[str, Any],
    ) -> dict[str, float] | None:
        if self._camera_info is None:
            return None
        point = self._deserialize_point(point_payload)
        if point is None:
            return None
        camera_frame_id = self._camera_info.header.frame_id
        if not camera_frame_id:
            return None
        try:
            camera_point = self._tf_buffer.transform(
                point,
                camera_frame_id,
                timeout=Duration(seconds=0.05),
            )
        except Exception as error:
            if self._should_log_projection_debug():
                self.get_logger().warning(
                    "Projection debug: failed to transform point for reprojection "
                    f"from {point.header.frame_id!r} to {camera_frame_id!r}: {error}"
                )
            return None
        forward_m = self._camera_forward_from_point(camera_point)
        normalized = self._normalized_pixel_from_camera_point(camera_point)
        if normalized is None:
            if self._should_log_projection_debug():
                self.get_logger().warning(
                    "Projection debug: point is behind camera for reprojection "
                    f"camera_frame_convention={self._camera_frame_convention}; "
                    f"camera_xyz=({camera_point.point.x:.4f}, "
                    f"{camera_point.point.y:.4f}, {camera_point.point.z:.4f})"
                )
            return None
        x_normalized, y_normalized = normalized
        fx = float(self._camera_info.k[0])
        fy = float(self._camera_info.k[4])
        cx = float(self._camera_info.k[2])
        cy = float(self._camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            if self._should_log_projection_debug():
                self.get_logger().warning(
                    "Projection debug: invalid intrinsics during reprojection "
                    f"fx={fx:.4f}; fy={fy:.4f}"
                )
            return None
        u = x_normalized * fx + cx
        v = y_normalized * fy + cy
        scale_x = float(self._image_size) / max(float(self._camera_info.width), 1.0)
        scale_y = float(self._image_size) / max(float(self._camera_info.height), 1.0)
        model_u = float(u * scale_x)
        model_v = float(v * scale_y)
        return {
            "model_u": model_u,
            "model_v": model_v,
            "center_x": model_u / max(float(self._image_size - 1), 1.0) * 2.0 - 1.0,
            "center_y": model_v / max(float(self._image_size - 1), 1.0) * 2.0 - 1.0,
            "depth_m": float(forward_m),
        }

    def _project_world_point_to_model_uv(
        self,
        point_payload: dict[str, Any],
    ) -> list[float] | None:
        measurement = self._project_world_point_to_model_measurement(point_payload)
        if measurement is None:
            return None
        return [measurement["model_u"], measurement["model_v"]]

    def _ground_truth_training_targets(self) -> list[dict[str, float]]:
        targets = []
        for item in self._ground_truth_items:
            point_payload = item.get("point")
            if not isinstance(point_payload, dict):
                continue
            measurement = self._project_world_point_to_model_measurement(point_payload)
            if measurement is None:
                continue
            if (
                measurement["model_u"] < 0.0
                or measurement["model_u"] > float(self._image_size - 1)
                or measurement["model_v"] < 0.0
                or measurement["model_v"] > float(self._image_size - 1)
            ):
                continue
            targets.append(measurement)
        targets.sort(key=lambda target: (target["model_v"], target["model_u"]))
        return targets[:NUM_QUERIES]

    def _cleanup_pending_candidates(self) -> None:
        now = time.monotonic()
        stale_ids = [
            candidate_id
            for candidate_id, payload in self._pending_candidates.items()
            if now - float(payload.get("created_monotonic", now))
            > self._pending_candidate_timeout_sec
        ]
        for candidate_id in stale_ids:
            self._pending_candidates.pop(candidate_id, None)

    def _should_log_selection_debug(self) -> bool:
        return self._frame_count == 0 or self._frame_count % self._log_every == 0

    def _should_log_projection_debug(self) -> bool:
        return self._online_training_enabled or self._should_log_selection_debug()

    def _render_query_alpha_mask(
        self,
        outputs: dict[str, torch.Tensor],
        query_index: int,
    ) -> torch.Tensor:
        patch_alpha = outputs["patch_alpha"][0, query_index : query_index + 1].float()
        centers = outputs["centers"][0, query_index : query_index + 1].float()
        sizes = outputs["sizes"][0, query_index : query_index + 1].float()
        dummy_depth = torch.ones(
            1,
            1,
            1,
            device=patch_alpha.device,
            dtype=patch_alpha.dtype,
        )
        dummy_depth_scale = torch.ones_like(dummy_depth)
        _identity, _depth, rendered_occupancy = render_billboards(
            patch_alpha=patch_alpha.unsqueeze(0),
            centers=centers.unsqueeze(0),
            depth=dummy_depth,
            sizes=sizes.unsqueeze(0),
            depth_scale=dummy_depth_scale,
            image_height=self._image_size,
            image_width=self._image_size,
        )
        return rendered_occupancy[0, 0].detach().float()

    def _billboard_world_point(
        self,
        query_alpha: torch.Tensor,
        predicted_depth: torch.Tensor,
        image_header,
        *,
        source_width: int,
        source_height: int,
    ) -> tuple[PointStamped, PointStamped, dict[str, float]] | None:
        if self._camera_info is None:
            return None
        alpha = query_alpha.detach().float()
        depth = predicted_depth.detach().float()
        if depth.ndim == 3:
            depth = depth[0]
        mask = alpha > self._query_presence_threshold
        if int(mask.sum().item()) < 4:
            mask = alpha > max(float(alpha.max().item()) * 0.5, 1e-4)
        if int(mask.sum().item()) < 4:
            return None

        y_indices, x_indices = torch.nonzero(mask, as_tuple=True)
        weights = alpha[y_indices, x_indices].clamp_min(1e-6)
        weight_sum = weights.sum()
        if float(weight_sum.item()) <= 0.0:
            return None
        model_u = float((x_indices.float() * weights).sum().item() / weight_sum.item())
        model_v = float((y_indices.float() * weights).sum().item() / weight_sum.item())
        forward_m = float((depth[y_indices, x_indices] * weights).sum().item() / weight_sum.item())
        if forward_m <= 0.0:
            return None

        scale_x = float(source_width) / float(self._image_size)
        scale_y = float(source_height) / float(self._image_size)
        u = model_u * scale_x
        v = model_v * scale_y

        fx = float(self._camera_info.k[0])
        fy = float(self._camera_info.k[4])
        cx = float(self._camera_info.k[2])
        cy = float(self._camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return None

        x_normalized = (u - cx) / fx
        y_normalized = (v - cy) / fy
        camera_x, camera_y, camera_z, forward_m = self._camera_point_from_pixel(
            x_normalized,
            y_normalized,
            forward_m,
        )

        camera_pose = PoseStamped()
        camera_pose.header.stamp = rclpy.time.Time().to_msg()
        camera_pose.header.frame_id = self._camera_info.header.frame_id or image_header.frame_id
        camera_pose.pose.position.x = camera_x
        camera_pose.pose.position.y = camera_y
        camera_pose.pose.position.z = camera_z
        camera_pose.pose.orientation.w = 1.0

        try:
            world_pose = self._tf_buffer.transform(
                camera_pose,
                self._target_frame_id,
                timeout=Duration(seconds=0.05),
            )
            moveit_point = PointStamped()
            moveit_point.header = world_pose.header
            moveit_point.point = world_pose.pose.position
            moveit_point = self._tf_buffer.transform(
                moveit_point,
                self._moveit_target_frame_id,
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return None

        world_point = PointStamped()
        world_point.header = world_pose.header
        world_point.point = world_pose.pose.position
        metadata = {
            "model_u": model_u,
            "model_v": model_v,
            "source_u": float(u),
            "source_v": float(v),
            "camera_x": float(camera_x),
            "camera_y": float(camera_y),
            "camera_z": float(camera_z),
            "camera_forward_m": float(forward_m),
            "alpha_area": float(mask.float().sum().item()),
            "alpha_weight": float(weight_sum.item()),
        }
        return world_point, moveit_point, metadata

    def _publish_suggested_item_markers(
        self,
        outputs: dict[str, torch.Tensor],
        image_header,
        *,
        source_width: int,
        source_height: int,
    ) -> None:
        patch_alpha = outputs["patch_alpha"][0].detach().float()
        alpha_scores = patch_alpha.mean(dim=(1, 2)).cpu()
        query_indices = torch.nonzero(
            alpha_scores > self._query_presence_threshold,
            as_tuple=False,
        ).flatten().tolist()
        query_indices.sort(key=lambda index: float(alpha_scores[index].item()), reverse=True)
        query_indices = query_indices[: self._max_suggested_markers]
        predicted_depth = outputs["predicted_depth"][0].detach().float()

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = self._target_frame_id
        delete_marker.header.stamp = image_header.stamp
        delete_marker.ns = "vision_suggested_items"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0
        for query_index in query_indices:
            query_alpha = self._render_query_alpha_mask(outputs, query_index)
            point_result = self._billboard_world_point(
                query_alpha,
                predicted_depth,
                image_header,
                source_width=source_width,
                source_height=source_height,
            )
            if point_result is None:
                continue
            world_point, _moveit_point, _metadata = point_result
            alpha_score = float(alpha_scores[query_index].item())
            red, green, blue = _value_to_rgb(alpha_score)
            marker = Marker()
            marker.header = world_point.header
            marker.ns = "vision_suggested_items"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = world_point.point
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.035
            marker.scale.y = 0.035
            marker.scale.z = 0.035
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.color.a = max(0.25, min(1.0, alpha_score))
            marker_array.markers.append(marker)
            marker_id += 1

        self._suggested_item_markers_publisher.publish(marker_array)

    def _select_candidate(
        self,
        outputs: dict[str, torch.Tensor],
        image_header,
        source_width: int,
        source_height: int,
        resized_rgb: np.ndarray,
    ) -> None:
        self._cleanup_pending_candidates()
        if self._camera_info is None:
            if self._should_log_selection_debug():
                self.get_logger().warning(
                    "Skipping candidate selection: no CameraInfo received yet."
                )
            return {"selected_query_index": None, "selected_mask": None}
        if len(self._pending_candidates) >= self._max_inflight_candidates:
            if self._should_log_selection_debug():
                self.get_logger().warning(
                    "Skipping candidate selection: inflight candidate limit reached "
                    f"({len(self._pending_candidates)}/{self._max_inflight_candidates})."
                )
            return {"selected_query_index": None, "selected_mask": None}

        patch_alpha = outputs["patch_alpha"][0].detach().float()
        alpha_scores = patch_alpha.mean(dim=(1, 2)).cpu()
        valid_query_indices = torch.nonzero(
            alpha_scores > self._query_presence_threshold,
            as_tuple=False,
        ).flatten()
        if valid_query_indices.numel() == 0:
            if self._should_log_selection_debug():
                self.get_logger().warning(
                    "Skipping candidate selection: no billboard passed alpha threshold "
                    f"{self._query_presence_threshold:.3f}. "
                    f"alpha_max={float(alpha_scores.max().item()):.3f} "
                    f"alpha_mean={float(alpha_scores.mean().item()):.3f}"
                )
            return {"selected_query_index": None, "selected_mask": None}

        effective_epsilon = (
            self._exploration_epsilon if self._online_training_enabled else 0.0
        )
        explored = random.random() < effective_epsilon
        if explored:
            query_index = int(random.choice(valid_query_indices.tolist()))
            selection_mode = "explore"
        else:
            query_index = int(
                max(valid_query_indices.tolist(), key=lambda index: float(alpha_scores[index].item()))
            )
            selection_mode = "greedy"
        selected_score = float(alpha_scores[query_index].item())
        query_value_probs = outputs["value_probs"][0, query_index].detach().float().cpu()
        value_left = float(query_value_probs[0].item())
        value_right = float(query_value_probs[1].item())
        selected_value = max(value_left, value_right)

        query_alpha = self._render_query_alpha_mask(outputs, query_index)
        point_result = self._billboard_world_point(
            query_alpha,
            outputs["predicted_depth"][0].detach().float(),
            image_header,
            source_width=source_width,
            source_height=source_height,
        )
        if point_result is None:
            if self._should_log_selection_debug():
                self.get_logger().warning(
                    "Skipping candidate selection: selected billboard could not be "
                    f"projected (query={query_index}, alpha_score={selected_score:.3f})."
                )
            return {"selected_query_index": query_index, "selected_mask": query_alpha.cpu().numpy()}

        world_point, moveit_point, projection = point_result
        stamp_ns = (
            int(world_point.header.stamp.sec) * 1_000_000_000
            + int(world_point.header.stamp.nanosec)
        )
        self._selection_serial += 1
        candidate_id = f"{stamp_ns}-{self._selection_serial:06d}-{query_index:02d}-auto"
        candidate_payload = {
            "candidate_id": candidate_id,
            "arm_side": "",
            "selection_mode": selection_mode,
            "query_index": int(query_index),
            "stamp_ns": stamp_ns,
            "presence": selected_score,
            "value_left": value_left,
            "value_right": value_right,
            "selected_value": selected_value,
            "model_uv": [projection["model_u"], projection["model_v"]],
            "image_uv": [projection["source_u"], projection["source_v"]],
            "camera_point_xyz": [
                projection["camera_x"],
                projection["camera_y"],
                projection["camera_z"],
            ],
            "camera_forward_m": projection["camera_forward_m"],
            "camera_frame_convention": self._camera_frame_convention,
            "billboard_alpha_score": selected_score,
            "billboard_alpha_area": projection["alpha_area"],
            "billboard_alpha_weight": projection["alpha_weight"],
            "world_point": self._serialize_point(world_point),
            "moveit_point": self._serialize_point(moveit_point),
        }
        gt_targets = self._ground_truth_training_targets()
        candidate_payload["ground_truth_target_count"] = len(gt_targets)
        reprojected_model_uv = self._project_world_point_to_model_uv(
            candidate_payload["world_point"]
        )
        reprojection_error_px = None
        if reprojected_model_uv is not None:
            reprojection_error_px = math.hypot(
                float(reprojected_model_uv[0]) - projection["model_u"],
                float(reprojected_model_uv[1]) - projection["model_v"],
            )
            candidate_payload["reprojected_model_uv"] = reprojected_model_uv
            candidate_payload["reprojection_error_px"] = float(reprojection_error_px)
            if self._writer is not None:
                self._writer.add_scalar(
                    "projection/reprojection_error_px",
                    float(reprojection_error_px),
                    self._frame_count,
                )
        self._selected_candidate_publisher.publish(_encode_payload(candidate_payload))
        self._pending_candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "arm_side": "auto",
            "arm_index": -1,
            "query_index": int(query_index),
            "selection_mode": selection_mode,
            "selected_value": selected_value,
            "value_left": value_left,
            "value_right": value_right,
            "presence": selected_score,
            "created_monotonic": time.monotonic(),
            "model_uv": [projection["model_u"], projection["model_v"]],
            "world_point": candidate_payload["world_point"],
            "gt_targets": gt_targets,
            "rgb": np.ascontiguousarray(resized_rgb.copy()),
        }
        if self._writer is not None:
            self._writer.add_scalar("selection/presence", selected_score, self._frame_count)
            self._writer.add_scalar("selection/selected_value", selected_value, self._frame_count)
            self._writer.add_scalar("selection/value_left", value_left, self._frame_count)
            self._writer.add_scalar("selection/value_right", value_right, self._frame_count)
            self._writer.add_scalar("selection/explored", 1.0 if explored else 0.0, self._frame_count)
        if self._should_log_projection_debug():
            reprojected_text = (
                "none"
                if reprojected_model_uv is None
                else f"({reprojected_model_uv[0]:.1f}, {reprojected_model_uv[1]:.1f})"
            )
            reprojection_error_text = (
                "none"
                if reprojection_error_px is None
                else f"{reprojection_error_px:.2f}"
            )
            self.get_logger().info(
                f"Selected billboard candidate {candidate_id} query={query_index} "
                f"mode={selection_mode} alpha_score={selected_score:.3f} "
                f"value_left={value_left:.3f} value_right={value_right:.3f} "
                f"moveit=({moveit_point.point.x:.3f}, {moveit_point.point.y:.3f}, {moveit_point.point.z:.3f}) "
                f"camera_frame_convention={self._camera_frame_convention} "
                f"gt_targets={len(gt_targets)}"
            )
            self.get_logger().info(
                "Projection debug: "
                f"image_header_frame={image_header.frame_id!r}; "
                f"camera_info_frame={self._camera_info.header.frame_id!r}; "
                f"source_size=({source_width}, {source_height}); "
                f"model_size={self._image_size}; "
                f"model_uv=({projection['model_u']:.1f}, {projection['model_v']:.1f}); "
                f"source_uv=({projection['source_u']:.1f}, {projection['source_v']:.1f}); "
                f"forward_m={projection['camera_forward_m']:.4f}; "
                f"camera_xyz=({projection['camera_x']:.4f}, "
                f"{projection['camera_y']:.4f}, {projection['camera_z']:.4f}); "
                f"world_xyz=({world_point.point.x:.4f}, {world_point.point.y:.4f}, {world_point.point.z:.4f}); "
                f"moveit_xyz=({moveit_point.point.x:.4f}, {moveit_point.point.y:.4f}, {moveit_point.point.z:.4f}); "
                f"reprojected_model_uv={reprojected_text}; "
                f"reprojection_error_px={reprojection_error_text}"
            )
        return {
            "selected_query_index": int(query_index),
            "selected_mask": query_alpha.cpu().numpy(),
        }

    def _handle_arm_state(self, message: String) -> None:
        payload = _decode_payload(message)
        if payload is None:
            return
        plan_outcome = payload.get("plan_outcome")
        if not isinstance(plan_outcome, dict):
            return
        request_id = str(plan_outcome.get("request_id", "")).strip()
        candidate_id = str(plan_outcome.get("candidate_id", "")).strip()
        step_name = str(plan_outcome.get("step_name", "")).strip().lower()
        if not request_id or not candidate_id or step_name != "approach pregrasp":
            return
        outcome_key = (candidate_id, request_id, step_name)
        if outcome_key in self._seen_plan_outcomes:
            return
        self._seen_plan_outcomes.add(outcome_key)
        pending = self._pending_candidates.pop(candidate_id, None)
        if pending is None:
            return

        success = 1.0 if bool(plan_outcome.get("success")) else 0.0
        outcome_arm_side = str(plan_outcome.get("arm_side") or pending["arm_side"]).strip()
        outcome_arm_index = _arm_side_to_index(outcome_arm_side)
        if outcome_arm_index is None:
            self.get_logger().warning(
                "Skipping value feedback replay sample with unknown arm side: "
                f"candidate_id={candidate_id}; request_id={request_id}; "
                f"arm_side={outcome_arm_side!r}"
            )
            return
        replay_item = {
            "rgb": pending["rgb"],
            "candidate_id": candidate_id,
            "request_id": request_id,
            "query_index": int(pending["query_index"]),
            "arm_index": int(outcome_arm_index),
            "label": float(success),
            "selection_mode": pending["selection_mode"],
            "selected_value": float(pending["selected_value"]),
            "value_left": float(pending.get("value_left", 0.0)),
            "value_right": float(pending.get("value_right", 0.0)),
            "presence": float(pending.get("presence", 0.0)),
            "arm_side": outcome_arm_side,
            "planning_time": float(plan_outcome.get("planning_time") or 0.0),
            "model_uv": pending.get("model_uv"),
            "gt_targets": list(pending.get("gt_targets") or []),
        }
        try:
            self._write_replay_sample(replay_item, candidate_id, request_id)
        except Exception as error:
            self.get_logger().warning(
                "Failed to write value feedback replay sample: "
                f"candidate_id={candidate_id}; request_id={request_id}; error={error}"
            )
        self._outcome_count += 1
        self.get_logger().info(
            "Pick feedback accepted: "
            f"candidate_id={candidate_id}; request_id={request_id}; "
            f"success={bool(success)}; "
            f"arm_side={outcome_arm_side}; arm_index={outcome_arm_index}; "
            f"query_index={replay_item['query_index']}; "
            f"checkpoint_step={self._train_step}; "
            f"gt_targets={len(replay_item['gt_targets'])}; "
            f"stored_model_uv={pending.get('model_uv')}; "
            f"world_point={pending.get('world_point')}"
        )

        if self._writer is not None:
            step = self._outcome_count
            self._writer.add_scalar("planner/approach_pregrasp_success", success, step)
            self._writer.add_scalar(
                f"planner/approach_pregrasp_success_{outcome_arm_side}",
                success,
                step,
            )
            self._writer.add_scalar(
                "planner/approach_pregrasp_planning_time",
                replay_item["planning_time"],
                step,
            )
            self._writer.add_scalar(
                "replay/buffer_size",
                len(self._replay_buffer),
                step,
            )
            self._writer.flush()

        self._log_training_mode_periodically()

    def _handle_reset(self, message: String) -> None:
        payload = _decode_payload(message)
        reset_reason = ""
        if isinstance(payload, dict):
            reset_reason = str(payload.get("reason", "")).strip()
        pending_count = len(self._pending_candidates)
        outcome_count = len(self._seen_plan_outcomes)
        self._pending_candidates.clear()
        self._seen_plan_outcomes.clear()
        self.get_logger().info(
            "Received vision reset"
            + (f" reason={reset_reason}" if reset_reason else "")
            + f"; cleared_pending={pending_count}; cleared_seen_outcomes={outcome_count}"
        )

    def _log_training_mode_periodically(self) -> None:
        if not self._online_training_enabled:
            return
        now = time.monotonic()
        if (
            self._last_training_mode_log_monotonic != 0.0
            and now - self._last_training_mode_log_monotonic < 30.0
        ):
            return
        self._last_training_mode_log_monotonic = now
        self.get_logger().info(
            "Vision mode: online-billboard-inference; "
            f"pending_candidates={len(self._pending_candidates)}; "
            f"checkpoint_step={self._train_step}; "
            f"checkpoint_reload_period_sec={self._checkpoint_reload_period_sec:.1f}"
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
        pixel_values = self._normalize_rgb_batch(resized_rgb[None, ...])

        start_time = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(
                device_type=self._device.type,
                enabled=self._use_mixed_precision and self._device.type == "cuda",
            ):
                outputs = self._model(pixel_values, render_outputs=True)
        inference_elapsed = time.perf_counter() - start_time

        self._publish_suggested_item_markers(
            outputs,
            message.header,
            source_width=message.width,
            source_height=message.height,
        )

        selection_debug = self._select_candidate(
            outputs,
            message.header,
            source_width=message.width,
            source_height=message.height,
            resized_rgb=resized_rgb,
        )
        if selection_debug is None:
            selection_debug = {
                "selected_query_index": None,
                "selected_mask": None,
            }
        debug_overlay = _compose_debug_panel(
            resized_rgb,
            outputs["predicted_identity"][0].detach().float(),
            outputs["predicted_depth"][0].detach().float(),
            outputs["predicted_occupancy"][0].detach().float(),
            outputs["value_probs"][0].detach().float(),
            selection_debug.get("selected_mask"),
        )
        self._debug_image_publisher.publish(
            _rgb_array_to_image_message(debug_overlay, header=message.header, encoding="rgb8")
        )

        self._frame_count += 1
        self._inference_time_sum += inference_elapsed
        if self._frame_count % DEFAULT_LOG_EVERY == 0:
            average_ms = 1000.0 * self._inference_time_sum / max(self._frame_count, 1)
            self.get_logger().info(
                "Published vision debug frame "
                f"count={self._frame_count}; "
                f"debug_size={debug_overlay.shape[1]}x{debug_overlay.shape[0]}; "
                f"avg_inference_ms={average_ms:.1f}"
            )
        if self._writer is not None:
            self._writer.add_scalar("inference/frame_time_sec", inference_elapsed, self._frame_count)

    def destroy_node(self) -> bool:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = VisionInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
