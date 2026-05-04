#!/usr/bin/env python3

import json
import math
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
from vision.model import HIDDEN_DIM, LATENT_DIM, NUM_QUERIES, create_query_model, model_device

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
DEFAULT_TARGET_FRAME_ID = "world"
DEFAULT_MOVEIT_TARGET_FRAME_ID = "yumi_body"
DEFAULT_CAMERA_FRAME_CONVENTION = "ros_optical"
DEFAULT_QUERY_PRESENCE_THRESHOLD = 0.2
DEFAULT_EXPLORATION_EPSILON = 0.10
DEFAULT_LOG_EVERY = 30
DEFAULT_REPLAY_BUFFER_CAPACITY = 2048
DEFAULT_MIN_REPLAY_SIZE = 2
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_TRAIN_STEPS_PER_TICK = 4
DEFAULT_TRAIN_TICK_PERIOD = 0.01
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GEOMETRY_LOSS_WEIGHT = 1.0
DEFAULT_PRESENCE_LOSS_WEIGHT = 0.2
DEFAULT_DEPTH_LOSS_WEIGHT = 0.2
DEFAULT_CHECKPOINT_SAVE_INTERVAL = 100
DEFAULT_PENDING_CANDIDATE_TIMEOUT = 30.0
DEFAULT_MAX_INFLIGHT_CANDIDATES = 1
DEFAULT_MAX_SUGGESTED_MARKERS = 32


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
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "value_in_slot_head": True,
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
        if checkpoint_path.exists():
            checkpoint_metadata = load_checkpoint(
                checkpoint_path,
                self._model,
                expected_config=self._checkpoint_config,
                map_location=self._device,
                strict=False,
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
        self._seen_plan_outcomes: set[str] = set()
        self._ground_truth_items: list[dict[str, Any]] = []
        self._outcome_count = 0
        self._last_training_mode_log_monotonic = 0.0
        self._last_replay_wait_log_size = 0
        self._checkpoint_reload_timer = self.create_timer(
            self._checkpoint_reload_period_sec,
            self._maybe_reload_checkpoint,
        )

        if checkpoint_path.exists():
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
        else:
            self.get_logger().warning(
                f"Checkpoint not found at {checkpoint_path}. Starting vision inference from random weights."
            )
        self.get_logger().info(
            "Vision inference ready "
            f"(image_topic={image_topic}, image_size={self._image_size}, device={self._device})"
        )
        self.get_logger().info(
            "Vision mode: "
            f"{'replay-recording' if self._online_training_enabled else 'inference-only'}; "
            f"candidate topic={selected_candidate_topic}; "
            f"suggested markers topic={suggested_item_markers_topic}; "
            f"arm state topic={arm_state_topic}"
        )
        self.get_logger().info(
            "Vision mode details: "
            f"exploration_epsilon="
            f"{self._exploration_epsilon if self._online_training_enabled else 0.0:.3f}; "
            f"replay_buffer_capacity={self._replay_buffer_capacity}; "
            f"min_replay_size={self._min_replay_size}; "
            f"train_batch_size={self._train_batch_size}; "
            f"geometry_loss_weight={self._geometry_loss_weight:.3f}; "
            f"presence_loss_weight={self._presence_loss_weight:.3f}; "
            f"depth_loss_weight={self._depth_loss_weight:.3f}; "
            f"replay_dir={self._replay_dir}; "
            f"camera_frame_convention={self._camera_frame_convention}"
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

    def _world_point_from_model_prediction(
        self,
        center: torch.Tensor,
        depth: torch.Tensor,
        image_header,
        *,
        source_width: int,
        source_height: int,
    ) -> PointStamped | None:
        if self._camera_info is None:
            return None
        model_u = float((center[0].item() + 1.0) * 0.5 * (self._image_size - 1))
        model_v = float((center[1].item() + 1.0) * 0.5 * (self._image_size - 1))
        forward_m = float(depth[0].item())
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
        camera_x, camera_y, camera_z, _forward_m = self._camera_point_from_pixel(
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
        except Exception:
            return None

        world_point = PointStamped()
        world_point.header = world_pose.header
        world_point.point = world_pose.pose.position
        return world_point

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

    def _build_debug_overlay(
        self,
        rgb: np.ndarray,
        centers: torch.Tensor,
        sizes: torch.Tensor,
        depth: torch.Tensor,
        presence_probs: torch.Tensor,
        value_probs: torch.Tensor,
        selected_query_index: int | None,
        selected_arm_side: str | None,
        ground_truth_items: list[dict[str, Any]],
    ) -> np.ndarray:
        overlay = np.ascontiguousarray(rgb.copy())
        image_height, image_width = overlay.shape[:2]
        for query_index in range(centers.shape[0]):
            center_x = float(centers[query_index, 0].item())
            center_y = float(centers[query_index, 1].item())
            pixel_x = int(round((center_x + 1.0) * 0.5 * (image_width - 1)))
            pixel_y = int(round((center_y + 1.0) * 0.5 * (image_height - 1)))
            radius = int(
                max(
                    3.0,
                    0.5
                    * max(image_width, image_height)
                    * max(
                        float(sizes[query_index, 0].item()),
                        float(sizes[query_index, 1].item()),
                    ),
                )
            )
            presence = float(presence_probs[query_index].item())
            left_value = float(value_probs[query_index, 0].item())
            right_value = float(value_probs[query_index, 1].item())
            predicted_depth = float(depth[query_index].item())
            if presence < self._query_presence_threshold:
                color = (100, 100, 100)
            else:
                color = (40, 140, 255)
            thickness = 3 if query_index == selected_query_index else 1
            draw_target = overlay
            if presence < 0.50:
                draw_target = overlay.copy()
            cv2.circle(draw_target, (pixel_x, pixel_y), radius, color, thickness, cv2.LINE_AA)
            cv2.circle(draw_target, (pixel_x, pixel_y), 2, color, -1, cv2.LINE_AA)
            label = (
                f"q{query_index} L:{left_value:.2f} R:{right_value:.2f} "
                f"P:{presence:.2f} Z:{predicted_depth:.2f}"
            )
            cv2.putText(
                draw_target,
                label,
                (pixel_x + 6, max(18, pixel_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
            if presence < 0.50:
                overlay = cv2.addWeighted(draw_target, 0.80, overlay, 0.20, 0.0)
        for item in ground_truth_items:
            point_payload = item.get("point")
            if not isinstance(point_payload, dict):
                continue
            model_uv = self._project_world_point_to_model_uv(point_payload)
            if model_uv is None:
                continue
            pixel_x = int(
                round(float(model_uv[0]) / max(self._image_size - 1, 1) * (image_width - 1))
            )
            pixel_y = int(
                round(float(model_uv[1]) / max(self._image_size - 1, 1) * (image_height - 1))
            )
            if pixel_x < 0 or pixel_x >= image_width or pixel_y < 0 or pixel_y >= image_height:
                continue
            color = (40, 240, 80)
            cv2.drawMarker(
                overlay,
                (pixel_x, pixel_y),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(overlay, (pixel_x, pixel_y), 7, color, 2, cv2.LINE_AA)
        if selected_query_index is not None and selected_arm_side is not None:
            cv2.putText(
                overlay,
                f"selected q{selected_query_index} arm={selected_arm_side}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return overlay

    def _publish_suggested_item_markers(
        self,
        outputs: dict[str, torch.Tensor],
        image_header,
        *,
        source_width: int,
        source_height: int,
    ) -> None:
        centers = outputs["centers"][0].detach().float()
        depth = outputs["depth"][0].detach().float()
        presence_probs = outputs["presence_probs"][0].detach().float().cpu().squeeze(-1)
        query_indices = torch.nonzero(
            presence_probs > self._query_presence_threshold,
            as_tuple=False,
        ).flatten().tolist()
        query_indices.sort(key=lambda index: float(presence_probs[index].item()), reverse=True)
        query_indices = query_indices[: self._max_suggested_markers]

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = self._target_frame_id
        delete_marker.header.stamp = image_header.stamp
        delete_marker.ns = "vision_suggested_items"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0
        for query_index in query_indices:
            world_point = self._world_point_from_model_prediction(
                centers[query_index],
                depth[query_index],
                image_header,
                source_width=source_width,
                source_height=source_height,
            )
            if world_point is None:
                continue
            presence = float(presence_probs[query_index].item())
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
            marker.color.r = 0.1
            marker.color.g = 0.35
            marker.color.b = 1.0
            marker.color.a = max(0.25, min(1.0, presence))
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
            return None
        if len(self._pending_candidates) >= self._max_inflight_candidates:
            if self._should_log_selection_debug():
                self.get_logger().warning(
                    "Skipping candidate selection: inflight candidate limit reached "
                    f"({len(self._pending_candidates)}/{self._max_inflight_candidates})."
                )
            return None

        centers = outputs["centers"][0].detach().float()
        sizes = outputs["sizes"][0].detach().float()
        depth = outputs["depth"][0].detach().float()
        presence_probs = outputs["presence_probs"][0].detach().float().cpu().squeeze(-1)
        value_probs = outputs["value_probs"][0].detach().float().cpu()
        valid_query_indices = torch.nonzero(
            presence_probs > self._query_presence_threshold,
            as_tuple=False,
        ).flatten()
        if valid_query_indices.numel() == 0:
            if self._should_log_selection_debug():
                self.get_logger().warning(
                    "Skipping candidate selection: no query passed presence threshold "
                    f"{self._query_presence_threshold:.3f}. "
                    f"presence_max={float(presence_probs.max().item()):.3f} "
                    f"presence_mean={float(presence_probs.mean().item()):.3f}"
                )
            return {
                "selected_query_index": None,
                "selected_arm_side": None,
                "centers": centers,
                "sizes": sizes,
                "depth": depth,
                "presence_probs": presence_probs,
                "value_probs": value_probs,
            }

        candidate_pairs: list[tuple[int, str, float]] = []
        for query_index in valid_query_indices.tolist():
            candidate_pairs.append((query_index, "left", float(value_probs[query_index, 0].item())))
            candidate_pairs.append((query_index, "right", float(value_probs[query_index, 1].item())))
        if not candidate_pairs:
            return

        effective_epsilon = (
            self._exploration_epsilon if self._online_training_enabled else 0.0
        )
        explored = random.random() < effective_epsilon
        if explored:
            query_index, arm_side, selected_value = random.choice(candidate_pairs)
            selection_mode = "explore"
        else:
            query_index, arm_side, selected_value = max(candidate_pairs, key=lambda item: item[2])
            selection_mode = "greedy"

        model_u = float((centers[query_index, 0].item() + 1.0) * 0.5 * (self._image_size - 1))
        model_v = float((centers[query_index, 1].item() + 1.0) * 0.5 * (self._image_size - 1))
        forward_m = float(depth[query_index, 0].item())
        if forward_m <= 0.0:
            if self._should_log_selection_debug():
                self.get_logger().warning(
                    "Skipping candidate selection: selected query has non-positive depth "
                    f"(query={query_index}, arm={arm_side}, depth={forward_m:.4f})."
                )
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
            if self._should_log_selection_debug():
                self.get_logger().warning(
                    "Skipping candidate selection: invalid camera intrinsics "
                    f"(fx={fx:.4f}, fy={fy:.4f})."
                )
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
        except Exception as error:
            if self._should_log_selection_debug():
                self.get_logger().warning(
                    "Skipping candidate selection: TF transform failed "
                    f"from {camera_pose.header.frame_id} to "
                    f"{self._target_frame_id}/{self._moveit_target_frame_id}: {error}"
                )
            return None

        world_point = PointStamped()
        world_point.header = world_pose.header
        world_point.point = world_pose.pose.position
        stamp_ns = (
            int(world_point.header.stamp.sec) * 1_000_000_000
            + int(world_point.header.stamp.nanosec)
        )
        self._selection_serial += 1
        candidate_id = (
            f"{stamp_ns}-{self._selection_serial:06d}-{query_index:02d}-{arm_side}"
        )
        candidate_payload = {
            "candidate_id": candidate_id,
            "arm_side": arm_side,
            "selection_mode": selection_mode,
            "query_index": int(query_index),
            "stamp_ns": stamp_ns,
            "presence": float(presence_probs[query_index].item()),
            "value_left": float(value_probs[query_index, 0].item()),
            "value_right": float(value_probs[query_index, 1].item()),
            "selected_value": float(selected_value),
            "model_center_xy": [
                float(centers[query_index, 0].item()),
                float(centers[query_index, 1].item()),
            ],
            "model_size_xy": [
                float(sizes[query_index, 0].item()),
                float(sizes[query_index, 1].item()),
            ],
            "image_uv": [float(u), float(v)],
            "camera_point_xyz": [
                float(camera_pose.pose.position.x),
                float(camera_pose.pose.position.y),
                float(camera_pose.pose.position.z),
            ],
            "camera_forward_m": float(forward_m),
            "camera_frame_convention": self._camera_frame_convention,
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
                float(reprojected_model_uv[0]) - model_u,
                float(reprojected_model_uv[1]) - model_v,
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
            "arm_side": arm_side,
            "arm_index": 0 if arm_side == "left" else 1,
            "query_index": int(query_index),
            "selection_mode": selection_mode,
            "selected_value": float(selected_value),
            "created_monotonic": time.monotonic(),
            "model_uv": [float(model_u), float(model_v)],
            "world_point": candidate_payload["world_point"],
            "gt_targets": gt_targets,
            "rgb": np.ascontiguousarray(resized_rgb.copy()),
        }
        if self._writer is not None:
            self._writer.add_scalar("selection/selected_value", float(selected_value), self._frame_count)
            self._writer.add_scalar("selection/explored", 1.0 if explored else 0.0, self._frame_count)
            self._writer.add_scalar(
                f"selection/{arm_side}_selected",
                1.0,
                self._frame_count,
            )
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
                f"Selected candidate {candidate_id} query={query_index} arm={arm_side} "
                f"mode={selection_mode} value={selected_value:.3f} "
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
                f"model_uv=({model_u:.1f}, {model_v:.1f}); "
                f"source_uv=({u:.1f}, {v:.1f}); "
                f"K=(fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}); "
                f"norm=({x_normalized:.4f}, {y_normalized:.4f}); "
                f"forward_m={forward_m:.4f}; "
                f"camera_xyz=({camera_x:.4f}, {camera_y:.4f}, {camera_z:.4f}); "
                f"world_xyz=({world_point.point.x:.4f}, {world_point.point.y:.4f}, {world_point.point.z:.4f}); "
                f"moveit_xyz=({moveit_point.point.x:.4f}, {moveit_point.point.y:.4f}, {moveit_point.point.z:.4f}); "
                f"reprojected_model_uv={reprojected_text}; "
                f"reprojection_error_px={reprojection_error_text}"
            )
        return {
            "selected_query_index": int(query_index),
            "selected_arm_side": arm_side,
            "centers": centers,
            "sizes": sizes,
            "depth": depth,
            "presence_probs": presence_probs,
            "value_probs": value_probs,
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
        if request_id in self._seen_plan_outcomes:
            return
        self._seen_plan_outcomes.add(request_id)
        pending = self._pending_candidates.pop(candidate_id, None)
        if pending is None:
            return

        success = 1.0 if bool(plan_outcome.get("success")) else 0.0
        replay_item = {
            "rgb": pending["rgb"],
            "query_index": int(pending["query_index"]),
            "arm_index": int(pending["arm_index"]),
            "label": float(success),
            "selection_mode": pending["selection_mode"],
            "selected_value": float(pending["selected_value"]),
            "arm_side": pending["arm_side"],
            "planning_time": float(plan_outcome.get("planning_time") or 0.0),
            "model_uv": pending.get("model_uv"),
            "gt_targets": list(pending.get("gt_targets") or []),
        }
        self._replay_buffer.append(replay_item)
        self._outcome_count += 1
        if self._online_training_enabled:
            self._write_replay_sample(replay_item, candidate_id, request_id)
        self.get_logger().info(
            "Replay label accepted: "
            f"candidate_id={candidate_id}; request_id={request_id}; "
            f"success={bool(success)}; "
            f"replay_size={len(self._replay_buffer)}/{self._min_replay_size}; "
            f"replay_recording={self._online_training_enabled}; "
            f"checkpoint_step={self._train_step}; "
            f"gt_targets={len(replay_item['gt_targets'])}; "
            f"stored_model_uv={pending.get('model_uv')}; "
            f"world_point={pending.get('world_point')}"
        )

        if self._writer is not None:
            step = self._outcome_count
            self._writer.add_scalar("planner/approach_pregrasp_success", success, step)
            self._writer.add_scalar(
                f"planner/approach_pregrasp_success_{pending['arm_side']}",
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
            "Vision mode: replay-recording; "
            f"replay_size={len(self._replay_buffer)}/{self._min_replay_size}; "
            f"pending_candidates={len(self._pending_candidates)}; "
            f"checkpoint_step={self._train_step}; "
            f"replay_dir={self._replay_dir}"
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
                outputs = self._model(pixel_values)
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
                "selected_arm_side": None,
                "centers": outputs["centers"][0].detach().float(),
                "sizes": outputs["sizes"][0].detach().float(),
                "depth": outputs["depth"][0].detach().float(),
                "presence_probs": outputs["presence_probs"][0].detach().float().cpu().squeeze(-1),
                "value_probs": outputs["value_probs"][0].detach().float().cpu(),
            }
        debug_overlay = self._build_debug_overlay(
            resized_rgb,
            selection_debug["centers"],
            selection_debug["sizes"],
            selection_debug["depth"],
            selection_debug["presence_probs"],
            selection_debug["value_probs"],
            selection_debug["selected_query_index"],
            selection_debug["selected_arm_side"],
            self._ground_truth_items,
        )
        self._debug_image_publisher.publish(
            _rgb_array_to_image_message(debug_overlay, header=message.header, encoding="rgb8")
        )

        self._frame_count += 1
        self._inference_time_sum += inference_elapsed
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
