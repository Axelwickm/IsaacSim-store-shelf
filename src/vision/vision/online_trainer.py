#!/usr/bin/env python3

import json
import os
import random
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
import torch
import torch.nn.functional as F

from vision.checkpoints import load_checkpoint, save_checkpoint
from vision.dataset import (
    RGB_MEAN,
    RGB_STD,
    TRAIN_SPLIT_THRESHOLD,
    StoreShelfVisionDataset,
    stable_sample_split,
)
from vision.inference import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_REPLAY_DIR,
    _parameter_bool,
)
from vision.model import (
    BILLBOARD_MAX_SIZE,
    BILLBOARD_MIN_SIZE,
    HIDDEN_DIM,
    IDENTITY_ALPHA_EPSILON,
    LATENT_DIM,
    NUM_QUERIES,
    PATCH_SIZE,
    create_query_model,
    model_device,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    _SUMMARY_WRITER_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None
    _SUMMARY_WRITER_IMPORT_ERROR = error


DEFAULT_CUDA_MEMORY_LOG_PERIOD = 30.0
DEFAULT_CHECKPOINT_SAVE_INTERVAL = 100
DEFAULT_DEPTH_LOSS_WEIGHT = 0.2
DEFAULT_GEOMETRY_LOSS_WEIGHT = 1.0
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MIN_REPLAY_SIZE = 2
DEFAULT_PRESENCE_LOSS_WEIGHT = 0.2
DEFAULT_REPLAY_BUFFER_CAPACITY = 512
DEFAULT_TENSORBOARD_DIR = Path("/workspace/tensorboard/vision")
DEFAULT_TRAIN_BATCH_SIZE = 2
DEFAULT_TRAIN_STEPS_PER_TICK = 1
DEFAULT_TRAIN_TICK_PERIOD = 0.05
DEFAULT_VALUE_LOSS_WEIGHT = 1.0
DEFAULT_TRAINING_DEBUG_PERIOD_STEPS = 25
DEFAULT_SAMPLE_PREFETCH_SIZE = 64
DEFAULT_SAMPLE_LOADER_WORKERS = 2
DEFAULT_EVAL_INTERVAL_STEPS = 2000


def _bytes_to_mib(byte_count: int | float) -> float:
    return float(byte_count) / (1024.0 * 1024.0)


def _rgb_tensor_to_display(rgb_tensor: torch.Tensor) -> np.ndarray:
    rgb = rgb_tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _identity_to_display(identity: torch.Tensor, foreground_mask: torch.Tensor | None = None) -> np.ndarray:
    identity_np = identity.detach().float().cpu().numpy()
    if identity_np.ndim == 3:
        identity_np = identity_np[0]
    hue = np.mod(identity_np * 137.508, 360.0) / 2.0
    saturation = np.full_like(hue, 180.0)
    value = np.full_like(hue, 255.0)
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.uint8)
    display = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    display[identity_np <= 0.0] = 0
    if foreground_mask is not None:
        mask = foreground_mask.detach().float().cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        display[mask <= 0.05] = 0
    return display


def _depth_to_display(
    depth: torch.Tensor,
    foreground_mask: torch.Tensor | None = None,
    *,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> np.ndarray:
    depth_np = depth.detach().float().cpu().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    valid = depth_np > 1e-6
    if foreground_mask is not None:
        mask_np = foreground_mask.detach().float().cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        valid = np.logical_and(valid, mask_np > 0.05)
    if min_depth is None or max_depth is None:
        if np.any(valid):
            valid_values = depth_np[valid]
            min_depth = float(np.percentile(valid_values, 5.0))
            max_depth = float(np.percentile(valid_values, 95.0))
        else:
            min_depth = 0.0
            max_depth = 1.0
    if max_depth <= min_depth:
        max_depth = min_depth + 1e-3
    normalized = np.clip((depth_np - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    display = np.repeat((normalized[..., None] * 255.0).astype(np.uint8), 3, axis=2)
    display[~valid] = 0
    return display


def _render_assigned_identity_for_display(
    patch_alpha: torch.Tensor,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    depths: torch.Tensor,
    slot_instance_ids: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    patch_alpha = patch_alpha.float()
    centers = centers.float()
    sizes = sizes.float()
    depths = depths.float()
    device = patch_alpha.device
    num_queries, patch_h, patch_w = patch_alpha.shape
    ys = torch.linspace(-1.0, 1.0, image_height, device=device)
    xs = torch.linspace(-1.0, 1.0, image_width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.view(1, image_height, image_width)
    grid_y = grid_y.view(1, image_height, image_width)

    centers_x = centers[:, 0].view(num_queries, 1, 1)
    centers_y = centers[:, 1].view(num_queries, 1, 1)
    sizes_x = sizes[:, 0].view(num_queries, 1, 1)
    sizes_y = sizes[:, 1].view(num_queries, 1, 1)
    local_x = (grid_x - centers_x) / sizes_x
    local_y = (grid_y - centers_y) / sizes_y
    sample_grid = torch.stack([local_x, local_y], dim=-1)
    sampled_alpha = torch.nn.functional.grid_sample(
        patch_alpha.unsqueeze(1),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)

    rendered_identity = torch.zeros(
        image_height,
        image_width,
        device=device,
        dtype=torch.float32,
    )
    remaining_identity = torch.ones(
        image_height,
        image_width,
        device=device,
        dtype=torch.float32,
    )
    query_order = torch.argsort(depths.flatten(), descending=False)
    for query_index in query_order.tolist():
        instance_id = int(slot_instance_ids[query_index].item())
        if instance_id <= 0:
            continue
        alpha = sampled_alpha[query_index]
        claim_strength = alpha * remaining_identity
        rendered_identity = torch.where(
            claim_strength > IDENTITY_ALPHA_EPSILON,
            torch.full_like(rendered_identity, float(instance_id)),
            rendered_identity,
        )
        remaining_identity = remaining_identity * (1.0 - alpha)
    return rendered_identity


def _draw_debug_lines(panel: np.ndarray, lines: list[str]) -> np.ndarray:
    display = np.ascontiguousarray(panel.copy())
    line_height = 17
    margin = 6
    box_height = margin * 2 + line_height * len(lines)
    overlay = display.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (display.shape[1], min(box_height, display.shape[0])),
        (0, 0, 0),
        thickness=-1,
    )
    display = cv2.addWeighted(overlay, 0.62, display, 0.38, 0.0)
    for index, line in enumerate(lines):
        y = margin + 12 + index * line_height
        cv2.putText(
            display,
            line,
            (margin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return display


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _sample_cpu_bytes(sample: dict[str, Any]) -> int:
    total = 0
    for value in sample.values():
        if isinstance(value, torch.Tensor):
            total += _tensor_bytes(value)
    return total


def _sample_file_signature(sample_dir: Path, sample_id: str) -> int | None:
    paths = [
        sample_dir / f"rgb_{sample_id}.png",
        sample_dir / f"instance_segmentation_{sample_id}.png",
        sample_dir / f"distance_to_camera_{sample_id}.npy",
        sample_dir / f"instance_segmentation_mapping_{sample_id}.json",
        sample_dir / f"instance_segmentation_semantics_mapping_{sample_id}.json",
    ]
    try:
        return max(path.stat().st_mtime_ns for path in paths)
    except FileNotFoundError:
        return None


def _feedback_file_signature(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _default_run_name() -> str:
    return "vision-online-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def _build_instance_targets(
    instance_targets: torch.Tensor,
    depth_targets: torch.Tensor,
    num_queries: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = instance_targets.shape[0]
    image_height = instance_targets.shape[-2]
    image_width = instance_targets.shape[-1]
    device = instance_targets.device
    target_instance_ids = torch.zeros(
        batch_size,
        num_queries,
        device=device,
        dtype=instance_targets.dtype,
    )
    target_centers = torch.zeros(batch_size, num_queries, 2, device=device)
    target_depths = torch.zeros(batch_size, num_queries, 1, device=device)
    target_sizes = torch.zeros(batch_size, num_queries, 2, device=device)

    for batch_index in range(batch_size):
        instance_ids = torch.unique(instance_targets[batch_index])
        instance_entries = []
        for instance_id_tensor in instance_ids:
            instance_id = int(instance_id_tensor.item())
            if instance_id <= 0:
                continue
            mask = instance_targets[batch_index] == instance_id
            area = int(mask.sum().item())
            if area <= 0:
                continue
            instance_entries.append((area, instance_id, mask))
        instance_entries.sort(key=lambda entry: entry[0], reverse=True)

        for query_index, (_area, instance_id, mask) in enumerate(
            instance_entries[:num_queries]
        ):
            y_indices, x_indices = torch.nonzero(mask, as_tuple=True)
            x_mean = x_indices.float().mean()
            y_mean = y_indices.float().mean()
            x_min = x_indices.float().min()
            x_max = x_indices.float().max()
            y_min = y_indices.float().min()
            y_max = y_indices.float().max()
            depth_values = depth_targets[batch_index, 0][mask]
            valid_depth = depth_values > 1e-6
            if torch.any(valid_depth):
                target_depth = depth_values[valid_depth].median()
            else:
                target_depth = torch.tensor(0.0, device=device)

            target_instance_ids[batch_index, query_index] = instance_id
            target_centers[batch_index, query_index, 0] = (
                x_mean / max(image_width - 1, 1) * 2.0 - 1.0
            )
            target_centers[batch_index, query_index, 1] = (
                y_mean / max(image_height - 1, 1) * 2.0 - 1.0
            )
            target_depths[batch_index, query_index, 0] = target_depth
            target_sizes[batch_index, query_index, 0] = (
                (x_max - x_min + 1.0) / max(image_width - 1, 1)
            ).clamp(BILLBOARD_MIN_SIZE, BILLBOARD_MAX_SIZE)
            target_sizes[batch_index, query_index, 1] = (
                (y_max - y_min + 1.0) / max(image_height - 1, 1)
            ).clamp(BILLBOARD_MIN_SIZE, BILLBOARD_MAX_SIZE)

    return target_instance_ids, target_centers, target_depths, target_sizes


def _unique_center_instance_ids(
    center_instance_ids: torch.Tensor,
    depths: torch.Tensor,
) -> torch.Tensor:
    unique_center_ids = torch.zeros_like(center_instance_ids)
    query_order = torch.argsort(depths.squeeze(-1), dim=1, descending=False)

    for batch_index in range(center_instance_ids.shape[0]):
        assigned_ids: set[int] = set()
        for query_index in query_order[batch_index].tolist():
            instance_id = int(center_instance_ids[batch_index, query_index].item())
            if instance_id <= 0 or instance_id in assigned_ids:
                continue
            unique_center_ids[batch_index, query_index] = instance_id
            assigned_ids.add(instance_id)

    return unique_center_ids


def _build_alpha_targets(
    instance_targets: torch.Tensor,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    depths: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_queries = centers.shape[:2]
    image_height = instance_targets.shape[-2]
    image_width = instance_targets.shape[-1]
    device = instance_targets.device

    center_x = ((centers[..., 0] + 1.0) * 0.5 * (image_width - 1)).round().long()
    center_y = ((centers[..., 1] + 1.0) * 0.5 * (image_height - 1)).round().long()
    center_x = center_x.clamp(0, image_width - 1)
    center_y = center_y.clamp(0, image_height - 1)
    batch_index = torch.arange(batch_size, device=device)[:, None].expand(
        -1,
        num_queries,
    )
    target_instance_ids = instance_targets[batch_index, center_y, center_x]
    target_instance_ids = _unique_center_instance_ids(target_instance_ids, depths)

    ys = torch.linspace(-1.0, 1.0, patch_size, device=device)
    xs = torch.linspace(-1.0, 1.0, patch_size, device=device)
    patch_grid_y, patch_grid_x = torch.meshgrid(ys, xs, indexing="ij")
    patch_grid_x = patch_grid_x.view(1, 1, patch_size, patch_size)
    patch_grid_y = patch_grid_y.view(1, 1, patch_size, patch_size)

    sample_x = centers[..., 0].view(batch_size, num_queries, 1, 1) + (
        patch_grid_x * sizes[..., 0].view(batch_size, num_queries, 1, 1)
    )
    sample_y = centers[..., 1].view(batch_size, num_queries, 1, 1) + (
        patch_grid_y * sizes[..., 1].view(batch_size, num_queries, 1, 1)
    )
    sample_grid = torch.stack([sample_x, sample_y], dim=-1).view(
        batch_size * num_queries, patch_size, patch_size, 2
    )

    target_masks = (
        instance_targets[:, None, :, :] == target_instance_ids[:, :, None, None]
    ).float()
    sampled_masks = torch.nn.functional.grid_sample(
        target_masks.view(batch_size * num_queries, 1, image_height, image_width),
        sample_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch_size, num_queries, patch_size, patch_size)

    valid_instance = (
        (target_instance_ids > 0).float().view(batch_size, num_queries, 1, 1)
    )
    return sampled_masks, valid_instance, target_instance_ids


class VisionOnlineTrainerNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_online_trainer")
        self.declare_parameter("checkpoint_dir", str(DEFAULT_CHECKPOINT_DIR))
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("replay_dir", str(DEFAULT_REPLAY_DIR))
        self.declare_parameter("image_size", 384)
        self.declare_parameter("use_mixed_precision", True)
        self.declare_parameter("replay_buffer_capacity", DEFAULT_REPLAY_BUFFER_CAPACITY)
        self.declare_parameter("min_replay_size", DEFAULT_MIN_REPLAY_SIZE)
        self.declare_parameter("train_batch_size", min(DEFAULT_TRAIN_BATCH_SIZE, 2))
        self.declare_parameter("train_split_threshold", TRAIN_SPLIT_THRESHOLD)
        self.declare_parameter("eval_interval_steps", DEFAULT_EVAL_INTERVAL_STEPS)
        self.declare_parameter("eval_batch_size", min(DEFAULT_TRAIN_BATCH_SIZE, 2))
        self.declare_parameter("train_steps_per_tick", DEFAULT_TRAIN_STEPS_PER_TICK)
        self.declare_parameter("train_tick_period_sec", DEFAULT_TRAIN_TICK_PERIOD)
        self.declare_parameter("learning_rate", DEFAULT_LEARNING_RATE)
        self.declare_parameter("geometry_loss_weight", DEFAULT_GEOMETRY_LOSS_WEIGHT)
        self.declare_parameter("presence_loss_weight", DEFAULT_PRESENCE_LOSS_WEIGHT)
        self.declare_parameter("depth_loss_weight", DEFAULT_DEPTH_LOSS_WEIGHT)
        self.declare_parameter("value_loss_weight", DEFAULT_VALUE_LOSS_WEIGHT)
        self.declare_parameter("freeze_backbone", True)
        self.declare_parameter("tensorboard_log_dir", str(DEFAULT_TENSORBOARD_DIR))
        self.declare_parameter("tensorboard_run_name", "")
        self.declare_parameter("checkpoint_save_interval", DEFAULT_CHECKPOINT_SAVE_INTERVAL)
        self.declare_parameter("replay_scan_period_sec", 1.0)
        self.declare_parameter("cuda_memory_log_period_sec", DEFAULT_CUDA_MEMORY_LOG_PERIOD)
        self.declare_parameter("training_debug_period_steps", DEFAULT_TRAINING_DEBUG_PERIOD_STEPS)
        self.declare_parameter("sample_prefetch_size", DEFAULT_SAMPLE_PREFETCH_SIZE)
        self.declare_parameter("sample_loader_workers", DEFAULT_SAMPLE_LOADER_WORKERS)

        checkpoint_dir = Path(str(self.get_parameter("checkpoint_dir").value)).resolve()
        checkpoint_path_value = str(self.get_parameter("checkpoint_path").value).strip()
        self._latest_checkpoint_path = checkpoint_dir / "latest.pt"
        checkpoint_path = (
            Path(checkpoint_path_value).resolve()
            if checkpoint_path_value
            else self._latest_checkpoint_path
        )
        self._replay_dir = Path(str(self.get_parameter("replay_dir").value)).resolve()
        self._image_size = int(self.get_parameter("image_size").value)
        self._use_mixed_precision = _parameter_bool(
            self.get_parameter("use_mixed_precision").value
        )
        self._sample_buffer_capacity = max(
            int(self.get_parameter("replay_buffer_capacity").value),
            1,
        )
        self._min_sample_size = max(int(self.get_parameter("min_replay_size").value), 1)
        self._train_batch_size = max(int(self.get_parameter("train_batch_size").value), 1)
        self._train_split_threshold = min(
            max(float(self.get_parameter("train_split_threshold").value), 0.0),
            1.0,
        )
        self._eval_interval_steps = max(
            int(self.get_parameter("eval_interval_steps").value),
            0,
        )
        self._eval_batch_size = max(int(self.get_parameter("eval_batch_size").value), 1)
        self._train_steps_per_tick = max(
            int(self.get_parameter("train_steps_per_tick").value),
            1,
        )
        self._train_tick_period_sec = max(
            float(self.get_parameter("train_tick_period_sec").value),
            0.001,
        )
        self._learning_rate = float(self.get_parameter("learning_rate").value)
        self._identity_loss_weight = float(
            self.get_parameter("geometry_loss_weight").value
        )
        self._missed_occupancy_loss_weight = float(
            self.get_parameter("presence_loss_weight").value
        )
        self._depth_loss_weight = float(self.get_parameter("depth_loss_weight").value)
        self._value_loss_weight = float(self.get_parameter("value_loss_weight").value)
        self._freeze_backbone = _parameter_bool(self.get_parameter("freeze_backbone").value)
        self._tensorboard_log_dir = Path(
            str(self.get_parameter("tensorboard_log_dir").value)
        ).resolve()
        self._tensorboard_run_name = str(
            self.get_parameter("tensorboard_run_name").value
        ).strip() or _default_run_name()
        self._tensorboard_run_dir = self._tensorboard_log_dir / self._tensorboard_run_name
        self._checkpoint_save_interval = max(
            int(self.get_parameter("checkpoint_save_interval").value),
            1,
        )
        sample_scan_period_sec = max(
            float(self.get_parameter("replay_scan_period_sec").value),
            0.1,
        )
        self._cuda_memory_log_period_sec = max(
            float(self.get_parameter("cuda_memory_log_period_sec").value),
            0.0,
        )
        self._training_debug_period_steps = max(
            int(self.get_parameter("training_debug_period_steps").value),
            1,
        )
        self._sample_prefetch_size = max(
            int(self.get_parameter("sample_prefetch_size").value),
            self._train_batch_size,
        )
        self._sample_loader_workers = max(
            int(self.get_parameter("sample_loader_workers").value),
            0,
        )

        if SummaryWriter is None:
            raise RuntimeError(
                "Online trainer requires TensorBoard logging, but "
                "torch.utils.tensorboard.SummaryWriter is unavailable. "
                f"Import error: {_SUMMARY_WRITER_IMPORT_ERROR!r}"
            )

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
        if self._freeze_backbone:
            for name, parameter in self._model.named_parameters():
                parameter.requires_grad = (
                    name.startswith("query_embed")
                    or name.startswith("decoder")
                    or name.startswith("slot_head")
                    or name.startswith("patch_decoder")
                )
        trainable_parameters = [
            parameter for parameter in self._model.parameters() if parameter.requires_grad
        ]
        self._optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self._learning_rate,
        )
        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self._use_mixed_precision and self._device.type == "cuda",
        )
        checkpoint_metadata = {"global_step": 0}
        if checkpoint_path.exists():
            try:
                checkpoint_metadata = load_checkpoint(
                    checkpoint_path,
                    self._model,
                    expected_config=self._checkpoint_config,
                    map_location=self._device,
                    strict=False,
                )
            except Exception as error:
                self.get_logger().warning(
                    f"Skipping incompatible online vision checkpoint {checkpoint_path}: {error}"
                )
        self._model.train()
        self._train_step = int(checkpoint_metadata["global_step"])
        self._dataset = StoreShelfVisionDataset(
            self._replay_dir,
            split="all",
            image_size=self._image_size,
            allow_empty=True,
        )
        self._sample_ids: list[str] = []
        self._train_sample_ids: list[str] = []
        self._test_sample_ids: list[str] = []
        self._sample_splits: dict[str, str] = {}
        self._sample_signatures: dict[str, int] = {}
        self._sample_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._sample_futures: dict[str, Future] = {}
        self._sample_executor = (
            ThreadPoolExecutor(
                max_workers=self._sample_loader_workers,
                thread_name_prefix="vision_sample_loader",
            )
            if self._sample_loader_workers > 0
            else None
        )
        self._feedback_buffer: deque[dict[str, Any]] = deque(maxlen=self._sample_buffer_capacity)
        self._test_feedback_buffer: deque[dict[str, Any]] = deque(
            maxlen=self._sample_buffer_capacity
        )
        self._loaded_feedback_signatures: dict[str, int] = {}
        self._sample_buffer_cpu_bytes = 0
        self._feedback_buffer_cpu_bytes = 0
        self._test_feedback_buffer_cpu_bytes = 0
        self._last_batch_gpu_bytes = 0
        self._writer = SummaryWriter(log_dir=str(self._tensorboard_run_dir))
        self._last_wait_log_size = -1
        self._debug_window_backend: str | None = None
        self._debug_window_failed = False
        self._debug_figure = None
        self._debug_axes = None
        self._debug_image_artist = None

        self._scan_timer = self.create_timer(sample_scan_period_sec, self._scan_training_inputs)
        self._train_timer = self.create_timer(
            self._train_tick_period_sec,
            self._handle_training_tick,
        )
        self._cuda_memory_timer = None
        if self._device.type == "cuda" and self._cuda_memory_log_period_sec > 0.0:
            self._cuda_memory_timer = self.create_timer(
                self._cuda_memory_log_period_sec,
                self._log_cuda_memory,
            )
        self._scan_training_inputs()

        self.get_logger().info(
            "Vision trainer ready "
            f"(pid={os.getpid()}, replay_dir={self._replay_dir}, "
            f"checkpoint={self._latest_checkpoint_path}, device={self._device}, "
            f"sample_capacity={self._sample_buffer_capacity}, "
            f"train_samples={len(self._train_sample_ids)}, "
            f"test_samples={len(self._test_sample_ids)}, "
            f"train_feedback={len(self._feedback_buffer)}, "
            f"test_feedback={len(self._test_feedback_buffer)}, "
            f"train_split_threshold={self._train_split_threshold:.3f}, "
            f"sample_prefetch={self._sample_prefetch_size}, "
            f"sample_loader_workers={self._sample_loader_workers}, "
            f"batch_size={self._train_batch_size}, steps_per_tick={self._train_steps_per_tick}, "
            f"eval_interval_steps={self._eval_interval_steps}, "
            f"value_loss_weight={self._value_loss_weight:.3f}, "
            f"training_debug_period_steps={self._training_debug_period_steps})"
        )
        self._log_cuda_memory()

    def _log_cuda_memory(self) -> None:
        if self._device.type != "cuda":
            return
        device = self._device
        self.get_logger().info(
            "Vision trainer CUDA memory: "
            f"pid={os.getpid()}; "
            f"allocated={_bytes_to_mib(torch.cuda.memory_allocated(device)):.1f}MiB; "
            f"reserved={_bytes_to_mib(torch.cuda.memory_reserved(device)):.1f}MiB; "
            f"max_allocated={_bytes_to_mib(torch.cuda.max_memory_allocated(device)):.1f}MiB; "
            f"active_batch_gpu={_bytes_to_mib(self._last_batch_gpu_bytes):.1f}MiB; "
            f"sample_buffer_cpu={_bytes_to_mib(self._sample_buffer_cpu_bytes):.1f}MiB; "
            f"samples={len(self._sample_cache)}/{len(self._sample_ids)}; "
            f"sample_loads_inflight={len(self._sample_futures)}; "
            f"feedback_buffer_cpu={_bytes_to_mib(self._feedback_buffer_cpu_bytes):.1f}MiB; "
            f"feedback={len(self._feedback_buffer)}; "
            f"test_feedback_buffer_cpu={_bytes_to_mib(self._test_feedback_buffer_cpu_bytes):.1f}MiB; "
            f"test_feedback={len(self._test_feedback_buffer)}"
        )

    def _scan_training_inputs(self) -> None:
        self._scan_sample_files()
        self._scan_feedback_dir()
        self._harvest_sample_futures()
        self._schedule_sample_prefetch()

    def _scan_sample_files(self) -> None:
        if not self._replay_dir.exists():
            return
        indexed_count = 0
        for path in sorted(self._replay_dir.glob("rgb_*.png")):
            sample_id = path.stem.split("_")[-1]
            signature = _sample_file_signature(self._replay_dir, sample_id)
            if signature is None:
                continue
            if self._sample_signatures.get(sample_id) == signature:
                continue
            self._sample_signatures[sample_id] = signature
            sample_split = stable_sample_split(path.name, self._train_split_threshold)
            previous_split = self._sample_splits.get(sample_id)
            if previous_split and previous_split != sample_split:
                if previous_split == "train" and sample_id in self._train_sample_ids:
                    self._train_sample_ids.remove(sample_id)
                if previous_split == "test" and sample_id in self._test_sample_ids:
                    self._test_sample_ids.remove(sample_id)
            self._sample_splits[sample_id] = sample_split
            if sample_id in self._sample_cache:
                removed = self._sample_cache.pop(sample_id)
                self._sample_buffer_cpu_bytes -= _sample_cpu_bytes(removed)
            if sample_id not in self._sample_ids:
                self._sample_ids.append(sample_id)
            if sample_split == "train" and sample_id not in self._train_sample_ids:
                self._train_sample_ids.append(sample_id)
            if sample_split == "test" and sample_id not in self._test_sample_ids:
                self._test_sample_ids.append(sample_id)
            indexed_count += 1
        if indexed_count:
            self.get_logger().info(
                f"Indexed {indexed_count} new/updated samples; "
                f"sample_index={len(self._sample_ids)}; "
                f"train={len(self._train_sample_ids)}; "
                f"test={len(self._test_sample_ids)}; "
                f"sample_cache={len(self._sample_cache)}/{self._sample_buffer_capacity}; "
                f"sample_cache_cpu={_bytes_to_mib(self._sample_buffer_cpu_bytes):.1f}MiB"
            )

    def _load_replicator_sample(self, sample_id: str) -> dict[str, Any]:
        return self._dataset.load_sample(sample_id)

    def _cache_sample(self, sample_id: str, sample: dict[str, Any]) -> None:
        if sample_id in self._sample_cache:
            previous = self._sample_cache.pop(sample_id)
            self._sample_buffer_cpu_bytes -= _sample_cpu_bytes(previous)
        while len(self._sample_cache) >= self._sample_buffer_capacity:
            _evicted_id, evicted = self._sample_cache.popitem(last=False)
            self._sample_buffer_cpu_bytes -= _sample_cpu_bytes(evicted)
        self._sample_cache[sample_id] = sample
        self._sample_buffer_cpu_bytes += _sample_cpu_bytes(sample)

    def _load_sample_sync(self, sample_id: str) -> dict[str, Any] | None:
        future = self._sample_futures.pop(sample_id, None)
        try:
            sample = future.result() if future is not None else self._load_replicator_sample(sample_id)
        except Exception as error:
            self.get_logger().warning(f"Skipping invalid vision sample {sample_id}: {error}")
            self._sample_signatures.pop(sample_id, None)
            return None
        self._cache_sample(sample_id, sample)
        return sample

    def _harvest_sample_futures(self) -> None:
        completed_ids = [
            sample_id
            for sample_id, future in self._sample_futures.items()
            if future.done()
        ]
        loaded_count = 0
        for sample_id in completed_ids:
            future = self._sample_futures.pop(sample_id)
            try:
                sample = future.result()
            except Exception as error:
                self.get_logger().warning(f"Skipping invalid vision sample {sample_id}: {error}")
                self._sample_signatures.pop(sample_id, None)
                continue
            self._cache_sample(sample_id, sample)
            loaded_count += 1
        if loaded_count:
            self.get_logger().info(
                f"Prefetched {loaded_count} samples; "
                f"sample_cache={len(self._sample_cache)}/{self._sample_buffer_capacity}; "
                f"sample_cache_cpu={_bytes_to_mib(self._sample_buffer_cpu_bytes):.1f}MiB"
            )

    def _schedule_sample_prefetch(self) -> None:
        if not self._train_sample_ids:
            return
        target_cache_size = min(
            self._sample_prefetch_size,
            self._sample_buffer_capacity,
            len(self._train_sample_ids),
        )
        train_cached_count = sum(
            1
            for sample_id in self._sample_cache.keys()
            if self._sample_splits.get(sample_id) == "train"
        )
        train_future_count = sum(
            1
            for sample_id in self._sample_futures.keys()
            if self._sample_splits.get(sample_id) == "train"
        )
        wanted = max(target_cache_size - train_cached_count - train_future_count, 0)
        if wanted <= 0:
            return
        shuffled_ids = list(self._train_sample_ids)
        random.shuffle(shuffled_ids)
        for sample_id in shuffled_ids:
            if wanted <= 0:
                break
            if sample_id in self._sample_cache or sample_id in self._sample_futures:
                continue
            if self._sample_executor is None:
                sample = self._load_sample_sync(sample_id)
                if sample is not None:
                    wanted -= 1
                continue
            self._sample_futures[sample_id] = self._sample_executor.submit(
                self._load_replicator_sample,
                sample_id,
            )
            wanted -= 1

    def _load_feedback_sample(self, path: Path) -> dict[str, Any] | None:
        try:
            with np.load(path, allow_pickle=False) as data:
                rgb = np.asarray(data["rgb"])
                metadata_json = data["metadata_json"]
                if isinstance(metadata_json, np.ndarray):
                    metadata_json = metadata_json.item()
                metadata = json.loads(str(metadata_json))
        except Exception as error:
            self.get_logger().warning(f"Skipping invalid feedback sample {path}: {error}")
            return None
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            self.get_logger().warning(f"Skipping feedback sample with invalid RGB shape {path}: {rgb.shape}")
            return None
        if rgb.shape[0] != self._image_size or rgb.shape[1] != self._image_size:
            rgb = cv2.resize(
                rgb[:, :, :3],
                (self._image_size, self._image_size),
                interpolation=cv2.INTER_AREA,
            )
        else:
            rgb = rgb[:, :, :3]
        try:
            query_index = int(metadata["query_index"])
            arm_index = int(metadata["arm_index"])
            label = float(metadata["label"])
        except (KeyError, TypeError, ValueError) as error:
            self.get_logger().warning(f"Skipping feedback sample with invalid metadata {path}: {error}")
            return None
        if not (0 <= query_index < NUM_QUERIES) or arm_index not in {0, 1}:
            self.get_logger().warning(
                "Skipping feedback sample with out-of-range target "
                f"{path}: query_index={query_index}, arm_index={arm_index}"
            )
            return None
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        normalized = (rgb_tensor - RGB_MEAN.view(3, 1, 1)) / RGB_STD.view(3, 1, 1)
        return {
            "rgb_normalized": normalized,
            "query_index": query_index,
            "arm_index": arm_index,
            "label": 1.0 if label >= 0.5 else 0.0,
            "path": str(path),
            "candidate_id": str(metadata.get("candidate_id", "")),
            "request_id": str(metadata.get("request_id", "")),
        }

    def _scan_feedback_dir(self) -> None:
        if not self._replay_dir.exists():
            return
        loaded_count = 0
        train_loaded_count = 0
        test_loaded_count = 0
        for path in sorted(self._replay_dir.glob("*.npz")):
            if path.name.endswith(".tmp.npz"):
                continue
            signature = _feedback_file_signature(path)
            if signature is None:
                continue
            path_key = str(path)
            if self._loaded_feedback_signatures.get(path_key) == signature:
                continue
            sample = self._load_feedback_sample(path)
            if sample is None:
                self._loaded_feedback_signatures[path_key] = signature
                continue
            sample_split = stable_sample_split(path.name, self._train_split_threshold)
            sample["split"] = sample_split
            sample_cpu_bytes = _sample_cpu_bytes(sample)
            if sample_split == "train":
                if len(self._feedback_buffer) == self._feedback_buffer.maxlen:
                    evicted = self._feedback_buffer[0]
                    self._feedback_buffer_cpu_bytes -= _sample_cpu_bytes(evicted)
                self._feedback_buffer.append(sample)
                self._feedback_buffer_cpu_bytes += sample_cpu_bytes
                train_loaded_count += 1
            else:
                if len(self._test_feedback_buffer) == self._test_feedback_buffer.maxlen:
                    evicted = self._test_feedback_buffer[0]
                    self._test_feedback_buffer_cpu_bytes -= _sample_cpu_bytes(evicted)
                self._test_feedback_buffer.append(sample)
                self._test_feedback_buffer_cpu_bytes += sample_cpu_bytes
                test_loaded_count += 1
            self._loaded_feedback_signatures[path_key] = signature
            loaded_count += 1
        if loaded_count:
            self.get_logger().info(
                f"Loaded {loaded_count} new/updated value feedback samples; "
                f"train_feedback={len(self._feedback_buffer)}/{self._sample_buffer_capacity} "
                f"(+{train_loaded_count}); "
                f"test_feedback={len(self._test_feedback_buffer)}/{self._sample_buffer_capacity} "
                f"(+{test_loaded_count}); "
                f"feedback_buffer_cpu={_bytes_to_mib(self._feedback_buffer_cpu_bytes):.1f}MiB; "
                f"test_feedback_buffer_cpu={_bytes_to_mib(self._test_feedback_buffer_cpu_bytes):.1f}MiB"
            )

    def _handle_training_tick(self) -> None:
        self._harvest_sample_futures()
        self._schedule_sample_prefetch()
        has_supervised_samples = len(self._train_sample_ids) >= self._min_sample_size
        if not has_supervised_samples:
            wait_log_size = (len(self._train_sample_ids), len(self._feedback_buffer))
            if wait_log_size != self._last_wait_log_size:
                self._last_wait_log_size = wait_log_size
                self.get_logger().info(
                    "Trainer waiting for supervised samples: "
                    f"replay_dir={self._replay_dir}; "
                    f"train_samples={len(self._train_sample_ids)}/{self._min_sample_size}; "
                    f"test_samples={len(self._test_sample_ids)}; "
                    f"train_feedback={len(self._feedback_buffer)}; "
                    f"test_feedback={len(self._test_feedback_buffer)}; "
                    f"sample_cache={len(self._sample_cache)}; "
                    f"inflight={len(self._sample_futures)}"
                )
            return
        for _ in range(self._train_steps_per_tick):
            metrics = self._run_one_train_step()
        if (
            self._eval_interval_steps > 0
            and self._train_step % self._eval_interval_steps < self._train_steps_per_tick
        ):
            self._run_test_evaluation()
        self._writer.flush()
        if self._train_step % max(self._checkpoint_save_interval, 1) < self._train_steps_per_tick:
            self.get_logger().info(
                "Online trainer step: "
                f"train_step={self._train_step}; "
                f"train_samples={len(self._train_sample_ids)}; "
                f"test_samples={len(self._test_sample_ids)}; "
                f"train_feedback={len(self._feedback_buffer)}; "
                f"test_feedback={len(self._test_feedback_buffer)}; "
                f"sample_cache={len(self._sample_cache)}; "
                f"loss={metrics['loss']:.6f}; "
                f"identity={metrics['identity_loss']:.6f}; "
                f"value={metrics['value_loss']:.6f}; "
                f"value_acc={metrics['value_accuracy']:.3f}; "
                f"feedback_batch={int(metrics['feedback_batch_size'])}; "
                f"depth={metrics['depth_loss']:.6f}; "
                f"gt_ids={metrics['debug/first_target_ids']:.0f}; "
                f"pred_ids={metrics['debug/first_pred_ids']:.0f}; "
                f"slot_targets={metrics['debug/slot_target_queries']:.0f}; "
                f"occ_frac={metrics['debug/pred_occ_frac']:.3f}; "
                f"alpha_mean={metrics['debug/patch_alpha_mean']:.3f}; "
                f"alpha_max={metrics['debug/patch_alpha_max']:.3f}; "
                f"size_px={metrics['debug/pred_size_mean_px']:.1f}/"
                f"{metrics['debug/pred_size_max_px']:.1f}; "
                f"target_size_px={metrics['debug/target_size_mean_px']:.1f}/"
                f"{metrics['debug/target_size_max_px']:.1f}; "
                f"grad_norm={metrics['grad_norm']:.2e}"
            )

    def _run_one_train_step(self) -> dict[str, float]:
        batch = self._sample_training_batch()
        if batch:
            pixel_values = torch.stack(
                [sample["rgb_normalized"] for sample in batch],
            ).to(self._device)
            identity_targets = torch.stack(
                [sample["instance_segmentation"] for sample in batch],
            ).to(self._device)
            depth_targets = torch.stack(
                [sample["distance_to_camera"] for sample in batch],
            ).to(self._device)
            self._last_batch_gpu_bytes = (
                _tensor_bytes(pixel_values)
                + _tensor_bytes(identity_targets)
                + _tensor_bytes(depth_targets)
            )
        else:
            pixel_values = None
            identity_targets = None
            depth_targets = None
            self._last_batch_gpu_bytes = 0
        feedback_batch = []
        if self._feedback_buffer:
            feedback_batch = random.sample(
                list(self._feedback_buffer),
                k=min(self._train_batch_size, len(self._feedback_buffer)),
            )
            feedback_pixel_values = torch.stack(
                [sample["rgb_normalized"] for sample in feedback_batch],
            ).to(self._device)
            feedback_query_indices = torch.tensor(
                [int(sample["query_index"]) for sample in feedback_batch],
                device=self._device,
                dtype=torch.long,
            )
            feedback_arm_indices = torch.tensor(
                [int(sample["arm_index"]) for sample in feedback_batch],
                device=self._device,
                dtype=torch.long,
            )
            feedback_labels = torch.tensor(
                [float(sample["label"]) for sample in feedback_batch],
                device=self._device,
                dtype=torch.float32,
            )
            self._last_batch_gpu_bytes += (
                _tensor_bytes(feedback_pixel_values)
                + _tensor_bytes(feedback_query_indices)
                + _tensor_bytes(feedback_arm_indices)
                + _tensor_bytes(feedback_labels)
            )
        else:
            feedback_pixel_values = None
            feedback_query_indices = None
            feedback_arm_indices = None
            feedback_labels = None
        if pixel_values is None:
            raise RuntimeError("No supervised samples available for training")

        self._optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self._device.type,
            enabled=self._use_mixed_precision and self._device.type == "cuda",
        ):
            if pixel_values is not None and identity_targets is not None and depth_targets is not None:
                gt_foreground = (identity_targets > 0).float().unsqueeze(1)
                (
                    target_instance_ids,
                    _target_centers,
                    _target_depths,
                    target_sizes,
                ) = _build_instance_targets(
                    identity_targets,
                    depth_targets,
                    NUM_QUERIES,
                )
                outputs = self._model(pixel_values, render_outputs=True)
                alpha_targets, alpha_target_weights, slot_instance_ids = _build_alpha_targets(
                    identity_targets,
                    outputs["centers"],
                    outputs["sizes"],
                    outputs["depth"],
                    outputs["patch_alpha"].shape[-1],
                )
                identity_loss = F.binary_cross_entropy_with_logits(
                    outputs["patch_alpha_logits"],
                    alpha_targets,
                    weight=alpha_target_weights.expand_as(alpha_targets),
                    reduction="sum",
                )
                identity_loss = identity_loss / alpha_target_weights.expand_as(
                    alpha_targets
                ).sum().clamp_min(1.0)
                zero_loss = identity_loss.detach() * 0.0
            else:
                gt_foreground = None
                target_instance_ids = None
                target_sizes = None
                outputs = None
                alpha_targets = None
                slot_instance_ids = None
                zero_loss = torch.zeros((), device=self._device)
                identity_loss = zero_loss
            center_loss = zero_loss
            size_loss = zero_loss
            slot_depth_loss = zero_loss
            if feedback_pixel_values is not None:
                feedback_outputs = self._model(feedback_pixel_values, render_outputs=False)
                batch_indices = torch.arange(
                    feedback_pixel_values.shape[0],
                    device=self._device,
                    dtype=torch.long,
                )
                selected_value_logits = feedback_outputs["value_logits"][
                    batch_indices,
                    feedback_query_indices,
                    feedback_arm_indices,
                ]
            else:
                selected_value_logits = None
        with torch.autocast(device_type=self._device.type, enabled=False):
            if outputs is not None and identity_targets is not None and depth_targets is not None:
                predicted_occupancy = outputs["predicted_occupancy"].float()
                predicted_depth = outputs["predicted_depth"].float()
                depth_error = F.smooth_l1_loss(
                    predicted_depth,
                    depth_targets.float(),
                    reduction="none",
                )
                depth_loss = (
                    depth_error * predicted_occupancy
                ).sum() / predicted_occupancy.sum().clamp_min(1e-6)
                missed_occupancy_loss = (
                    gt_foreground.float() * (1.0 - predicted_occupancy)
                ).mean()
                extra_occupancy_loss = (
                    (1.0 - gt_foreground.float()) * predicted_occupancy
                ).mean()
                debug_metrics = self._collect_training_debug_metrics(
                    identity_targets,
                    target_instance_ids,
                    slot_instance_ids,
                    target_sizes,
                    alpha_targets,
                    outputs,
                )
            else:
                depth_loss = zero_loss
                missed_occupancy_loss = zero_loss
                extra_occupancy_loss = zero_loss
                debug_metrics = self._empty_debug_metrics()
            loss = (
                self._identity_loss_weight * identity_loss
                + self._depth_loss_weight * depth_loss
                + self._missed_occupancy_loss_weight * missed_occupancy_loss
            )
            if selected_value_logits is not None and feedback_labels is not None:
                value_loss = F.binary_cross_entropy_with_logits(
                    selected_value_logits.float(),
                    feedback_labels.float(),
                )
                loss = loss + self._value_loss_weight * value_loss
                value_predictions = selected_value_logits.float().sigmoid() >= 0.5
                value_accuracy = (
                    value_predictions == (feedback_labels.float() >= 0.5)
                ).float().mean()
            else:
                value_loss = loss.detach() * 0.0
                value_accuracy = loss.detach() * 0.0

        self._scaler.scale(loss).backward()
        debug_metrics["grad_norm"] = self._grad_norm()
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._train_step += 1
        metrics = {
            "loss": float(loss.detach().cpu()),
            "identity_loss": float(identity_loss.detach().cpu()),
            "center_loss": float(center_loss.detach().cpu()),
            "size_loss": float(size_loss.detach().cpu()),
            "depth_loss": float(depth_loss.detach().cpu()),
            "slot_depth_loss": float(slot_depth_loss.detach().cpu()),
            "missed_occupancy_loss": float(missed_occupancy_loss.detach().cpu()),
            "extra_occupancy_loss": float(extra_occupancy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "value_accuracy": float(value_accuracy.detach().cpu()),
            "feedback_batch_size": float(len(feedback_batch)),
            **debug_metrics,
        }
        self._write_metrics(metrics)
        if batch and outputs is not None and slot_instance_ids is not None:
            self._maybe_show_training_debug(batch[0], outputs, metrics, slot_instance_ids)
        if self._train_step % self._checkpoint_save_interval == 0:
            self._save_checkpoint()
        return metrics

    def _run_test_evaluation(self) -> dict[str, float] | None:
        batch = self._sample_test_batch()
        feedback_batch = self._sample_test_feedback_batch()
        if not batch:
            if not feedback_batch:
                self._writer.add_scalar("eval/test_batch_size", 0, self._train_step)
                self._writer.add_scalar(
                    "eval/test_feedback_batch_size",
                    0,
                    self._train_step,
                )
                self.get_logger().info(
                    "Skipping vision test evaluation because no held-out samples are available"
                )
                return None
            metrics = self._run_feedback_evaluation(feedback_batch)
            self.get_logger().info(
                f"Vision feedback test evaluation: train_step={self._train_step}; "
                f"test_feedback_batch={len(feedback_batch)}/{len(self._test_feedback_buffer)}; "
                f"value_loss={metrics['value_loss']:.6f}; "
                f"value_acc={metrics['value_accuracy']:.3f}"
            )
            return metrics

        pixel_values = torch.stack(
            [sample["rgb_normalized"] for sample in batch],
        ).to(self._device)
        identity_targets = torch.stack(
            [sample["instance_segmentation"] for sample in batch],
        ).to(self._device)
        depth_targets = torch.stack(
            [sample["distance_to_camera"] for sample in batch],
        ).to(self._device)
        gt_foreground = (identity_targets > 0).float().unsqueeze(1)
        (
            target_instance_ids,
            _target_centers,
            _target_depths,
            target_sizes,
        ) = _build_instance_targets(
            identity_targets,
            depth_targets,
            NUM_QUERIES,
        )

        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad(), torch.autocast(
                device_type=self._device.type,
                enabled=self._use_mixed_precision and self._device.type == "cuda",
            ):
                outputs = self._model(pixel_values, render_outputs=True)
                alpha_targets, alpha_target_weights, slot_instance_ids = _build_alpha_targets(
                    identity_targets,
                    outputs["centers"],
                    outputs["sizes"],
                    outputs["depth"],
                    outputs["patch_alpha"].shape[-1],
                )
                identity_loss = F.binary_cross_entropy_with_logits(
                    outputs["patch_alpha_logits"],
                    alpha_targets,
                    weight=alpha_target_weights.expand_as(alpha_targets),
                    reduction="sum",
                )
                identity_loss = identity_loss / alpha_target_weights.expand_as(
                    alpha_targets
                ).sum().clamp_min(1.0)
            with torch.no_grad(), torch.autocast(device_type=self._device.type, enabled=False):
                predicted_occupancy = outputs["predicted_occupancy"].float()
                predicted_depth = outputs["predicted_depth"].float()
                depth_error = F.smooth_l1_loss(
                    predicted_depth,
                    depth_targets.float(),
                    reduction="none",
                )
                depth_loss = (
                    depth_error * predicted_occupancy
                ).sum() / predicted_occupancy.sum().clamp_min(1e-6)
                missed_occupancy_loss = (
                    gt_foreground.float() * (1.0 - predicted_occupancy)
                ).mean()
                extra_occupancy_loss = (
                    (1.0 - gt_foreground.float()) * predicted_occupancy
                ).mean()
                loss = (
                    self._identity_loss_weight * identity_loss
                    + self._depth_loss_weight * depth_loss
                    + self._missed_occupancy_loss_weight * missed_occupancy_loss
                )
                debug_metrics = self._collect_training_debug_metrics(
                    identity_targets,
                    target_instance_ids,
                    slot_instance_ids,
                    target_sizes,
                    alpha_targets,
                    outputs,
                )
        finally:
            if was_training:
                self._model.train()

        metrics = {
            "loss": float(loss.detach().cpu()),
            "identity_loss": float(identity_loss.detach().cpu()),
            "depth_loss": float(depth_loss.detach().cpu()),
            "missed_occupancy_loss": float(missed_occupancy_loss.detach().cpu()),
            "extra_occupancy_loss": float(extra_occupancy_loss.detach().cpu()),
            "value_loss": 0.0,
            "value_accuracy": 0.0,
            **debug_metrics,
        }
        if feedback_batch:
            feedback_metrics = self._run_feedback_evaluation(feedback_batch, write_metrics=False)
            metrics["loss"] += feedback_metrics["value_loss"]
            metrics["value_loss"] = feedback_metrics["value_loss"]
            metrics["value_accuracy"] = feedback_metrics["value_accuracy"]
        self._write_eval_metrics(metrics, len(batch))
        self.get_logger().info(
            f"Vision test evaluation: train_step={self._train_step}; "
            f"test_batch={len(batch)}/{len(self._test_sample_ids)}; "
            f"test_feedback_batch={len(feedback_batch)}/{len(self._test_feedback_buffer)}; "
            f"loss={metrics['loss']:.6f}; "
            f"identity={metrics['identity_loss']:.6f}; "
            f"value={metrics['value_loss']:.6f}; "
            f"value_acc={metrics['value_accuracy']:.3f}; "
            f"depth={metrics['depth_loss']:.6f}; "
            f"gt_ids={metrics['debug/first_target_ids']:.0f}; "
            f"pred_ids={metrics['debug/first_pred_ids']:.0f}; "
            f"occ_frac={metrics['debug/pred_occ_frac']:.3f}"
        )
        return metrics

    def _run_feedback_evaluation(
        self,
        feedback_batch: list[dict[str, Any]],
        *,
        write_metrics: bool = True,
    ) -> dict[str, float]:
        feedback_pixel_values = torch.stack(
            [sample["rgb_normalized"] for sample in feedback_batch],
        ).to(self._device)
        feedback_query_indices = torch.tensor(
            [int(sample["query_index"]) for sample in feedback_batch],
            device=self._device,
            dtype=torch.long,
        )
        feedback_arm_indices = torch.tensor(
            [int(sample["arm_index"]) for sample in feedback_batch],
            device=self._device,
            dtype=torch.long,
        )
        feedback_labels = torch.tensor(
            [float(sample["label"]) for sample in feedback_batch],
            device=self._device,
            dtype=torch.float32,
        )
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad(), torch.autocast(
                device_type=self._device.type,
                enabled=self._use_mixed_precision and self._device.type == "cuda",
            ):
                feedback_outputs = self._model(feedback_pixel_values, render_outputs=False)
                batch_indices = torch.arange(
                    feedback_pixel_values.shape[0],
                    device=self._device,
                    dtype=torch.long,
                )
                selected_value_logits = feedback_outputs["value_logits"][
                    batch_indices,
                    feedback_query_indices,
                    feedback_arm_indices,
                ]
            with torch.no_grad(), torch.autocast(device_type=self._device.type, enabled=False):
                value_loss = F.binary_cross_entropy_with_logits(
                    selected_value_logits.float(),
                    feedback_labels.float(),
                )
                value_accuracy = (
                    (selected_value_logits.float().sigmoid() >= 0.5)
                    == (feedback_labels.float() >= 0.5)
                ).float().mean()
        finally:
            if was_training:
                self._model.train()

        metrics = {
            "loss": float(value_loss.detach().cpu()),
            "identity_loss": 0.0,
            "depth_loss": 0.0,
            "missed_occupancy_loss": 0.0,
            "extra_occupancy_loss": 0.0,
            "value_loss": float(value_loss.detach().cpu()),
            "value_accuracy": float(value_accuracy.detach().cpu()),
            **self._empty_debug_metrics(),
        }
        if write_metrics:
            self._write_eval_metrics(metrics, 0)
        return metrics

    def _sample_test_batch(self) -> list[dict[str, Any]]:
        if not self._test_sample_ids:
            return []
        selected_ids = random.sample(
            self._test_sample_ids,
            k=min(self._eval_batch_size, len(self._test_sample_ids)),
        )
        batch = []
        for sample_id in selected_ids:
            sample = self._sample_cache.get(sample_id)
            if sample is None:
                sample = self._load_sample_sync(sample_id)
            if sample is None:
                continue
            self._sample_cache.move_to_end(sample_id)
            batch.append(sample)
        return batch

    def _sample_test_feedback_batch(self) -> list[dict[str, Any]]:
        if not self._test_feedback_buffer:
            return []
        return random.sample(
            list(self._test_feedback_buffer),
            k=min(self._eval_batch_size, len(self._test_feedback_buffer)),
        )

    def _sample_training_batch(self) -> list[dict[str, Any]]:
        cached_ids = [
            sample_id
            for sample_id in self._sample_cache.keys()
            if self._sample_splits.get(sample_id) == "train"
        ]
        batch_size = min(self._train_batch_size, len(self._train_sample_ids))
        selected_ids: list[str] = []
        if cached_ids:
            selected_ids.extend(random.sample(cached_ids, k=min(batch_size, len(cached_ids))))
        if len(selected_ids) < batch_size:
            uncached_ids = [
                sample_id
                for sample_id in self._train_sample_ids
                if sample_id not in selected_ids
            ]
            random.shuffle(uncached_ids)
            selected_ids.extend(uncached_ids[: batch_size - len(selected_ids)])

        batch = []
        for sample_id in selected_ids:
            sample = self._sample_cache.get(sample_id)
            if sample is None:
                sample = self._load_sample_sync(sample_id)
            if sample is None:
                continue
            self._sample_cache.move_to_end(sample_id)
            batch.append(sample)
        if len(batch) < max(1, min(batch_size, self._train_batch_size)):
            raise RuntimeError(
                "No loadable samples available for training "
                f"(train_samples={len(self._train_sample_ids)}, cache={len(self._sample_cache)})"
            )
        self._schedule_sample_prefetch()
        return batch

    def _grad_norm(self) -> float:
        total_sq = 0.0
        for parameter in self._model.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            total_sq += float(torch.sum(grad * grad).cpu())
        return total_sq ** 0.5

    def _empty_debug_metrics(self) -> dict[str, float]:
        return {
            "debug/target_queries": 0.0,
            "debug/slot_target_queries": 0.0,
            "debug/first_target_ids": 0.0,
            "debug/first_pred_ids": 0.0,
            "debug/gt_fg_frac": 0.0,
            "debug/pred_occ_mean": 0.0,
            "debug/pred_occ_frac": 0.0,
            "debug/patch_alpha_mean": 0.0,
            "debug/patch_alpha_max": 0.0,
            "debug/patch_alpha_area_frac": 0.0,
            "debug/alpha_target_area_frac": 0.0,
            "debug/pred_size_mean_px": 0.0,
            "debug/pred_size_min_px": 0.0,
            "debug/pred_size_max_px": 0.0,
            "debug/target_size_mean_px": 0.0,
            "debug/target_size_max_px": 0.0,
            "debug/presence_prob_mean": 0.0,
            "debug/value_prob_mean": 0.0,
        }

    def _collect_training_debug_metrics(
        self,
        identity_targets: torch.Tensor,
        target_instance_ids: torch.Tensor,
        slot_instance_ids: torch.Tensor,
        target_sizes: torch.Tensor,
        alpha_targets: torch.Tensor,
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        with torch.no_grad():
            gt_foreground = identity_targets > 0
            pred_occupancy = outputs["predicted_occupancy"].detach().float()
            pred_identity = outputs["predicted_identity"].detach().float()
            patch_alpha = outputs["patch_alpha"].detach().float()
            pred_sizes = outputs["sizes"].detach().float()
            valid_targets = target_instance_ids > 0
            valid_slots = slot_instance_ids > 0

            first_pred_labels = torch.unique(pred_identity[0][pred_identity[0] > 0.0])
            first_target_labels = torch.unique(identity_targets[0][identity_targets[0] > 0])
            pred_occ_mask = pred_occupancy > 0.05
            alpha_target_positive = alpha_targets > 0.5

            if torch.any(valid_targets):
                target_size_values = target_sizes[valid_targets]
                target_size_mean_px = float(target_size_values.mean().cpu()) * self._image_size
                target_size_max_px = float(target_size_values.max().cpu()) * self._image_size
            else:
                target_size_mean_px = 0.0
                target_size_max_px = 0.0

            return {
                "debug/target_queries": float(valid_targets.float().sum().cpu()),
                "debug/slot_target_queries": float(valid_slots.float().sum().cpu()),
                "debug/first_target_ids": float(first_target_labels.numel()),
                "debug/first_pred_ids": float(first_pred_labels.numel()),
                "debug/gt_fg_frac": float(gt_foreground.float().mean().cpu()),
                "debug/pred_occ_mean": float(pred_occupancy.mean().cpu()),
                "debug/pred_occ_frac": float(pred_occ_mask.float().mean().cpu()),
                "debug/patch_alpha_mean": float(patch_alpha.mean().cpu()),
                "debug/patch_alpha_max": float(patch_alpha.max().cpu()),
                "debug/patch_alpha_area_frac": float((patch_alpha > 0.5).float().mean().cpu()),
                "debug/alpha_target_area_frac": float(alpha_target_positive.float().mean().cpu()),
                "debug/pred_size_mean_px": float(pred_sizes.mean().cpu()) * self._image_size,
                "debug/pred_size_min_px": float(pred_sizes.min().cpu()) * self._image_size,
                "debug/pred_size_max_px": float(pred_sizes.max().cpu()) * self._image_size,
                "debug/target_size_mean_px": target_size_mean_px,
                "debug/target_size_max_px": target_size_max_px,
                "debug/presence_prob_mean": float(outputs["presence_probs"].detach().float().mean().cpu()),
                "debug/value_prob_mean": float(outputs["value_probs"].detach().float().mean().cpu()),
            }

    def _maybe_show_training_debug(
        self,
        sample: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        metrics: dict[str, float],
        slot_instance_ids: torch.Tensor,
    ) -> None:
        if self._train_step % self._training_debug_period_steps != 0:
            return
        gt_identity = sample["instance_segmentation"]
        gt_depth = sample["distance_to_camera"]
        gt_foreground = gt_identity > 0
        gt_depth_np = gt_depth.detach().float().cpu().numpy()
        gt_valid = np.logical_and(gt_depth_np[0] > 1e-6, gt_foreground.detach().cpu().numpy())
        if np.any(gt_valid):
            valid_depths = gt_depth_np[0][gt_valid]
            min_depth = float(np.percentile(valid_depths, 5.0))
            max_depth = float(np.percentile(valid_depths, 95.0))
        else:
            min_depth = None
            max_depth = None

        gt_identity_display = _identity_to_display(gt_identity)
        assigned_identity = _render_assigned_identity_for_display(
            outputs["patch_alpha"][0],
            outputs["centers"][0],
            outputs["sizes"][0],
            outputs["depth"][0],
            slot_instance_ids[0],
            gt_identity.shape[-2],
            gt_identity.shape[-1],
        )
        pred_identity_display = _identity_to_display(
            assigned_identity,
            outputs["predicted_occupancy"][0],
        )
        gt_depth_display = _depth_to_display(
            gt_depth,
            gt_foreground,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        pred_depth_display = _depth_to_display(
            outputs["predicted_depth"][0],
            outputs["predicted_occupancy"][0],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        top = np.concatenate([gt_identity_display, pred_identity_display], axis=1)
        bottom = np.concatenate([gt_depth_display, pred_depth_display], axis=1)
        panel = np.concatenate([top, bottom], axis=0)
        panel = _draw_debug_lines(
            panel,
            [
                f"step={self._train_step} loss={metrics['loss']:.3f} id={metrics['identity_loss']:.3f} depth={metrics['depth_loss']:.3f} val={metrics['value_loss']:.3f}",
                f"gt_ids={metrics['debug/first_target_ids']:.0f} pred_ids={metrics['debug/first_pred_ids']:.0f} target_q={metrics['debug/target_queries']:.0f} slot_q={metrics['debug/slot_target_queries']:.0f} grad={metrics['grad_norm']:.2e}",
                f"occ_mean={metrics['debug/pred_occ_mean']:.3f} occ_frac={metrics['debug/pred_occ_frac']:.3f} gt_fg={metrics['debug/gt_fg_frac']:.3f}",
                f"alpha_mean={metrics['debug/patch_alpha_mean']:.3f} alpha_max={metrics['debug/patch_alpha_max']:.3f} alpha_area={metrics['debug/patch_alpha_area_frac']:.3f} tgt_area={metrics['debug/alpha_target_area_frac']:.3f}",
                f"size_px pred={metrics['debug/pred_size_mean_px']:.1f}/{metrics['debug/pred_size_max_px']:.1f} target={metrics['debug/target_size_mean_px']:.1f}/{metrics['debug/target_size_max_px']:.1f}",
            ],
        )
        self._show_training_debug_window(panel)

    def _show_training_debug_window(self, panel: np.ndarray) -> None:
        if self._debug_window_failed:
            return
        if self._debug_window_backend in {None, "opencv"}:
            try:
                if self._debug_window_backend is None:
                    cv2.namedWindow("vision training progress", cv2.WINDOW_NORMAL)
                    self._debug_window_backend = "opencv"
                cv2.imshow("vision training progress", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                return
            except cv2.error as error:
                if self._debug_window_backend == "opencv":
                    self.get_logger().warning(
                        f"OpenCV training visualization window failed: {error}"
                    )
                self._debug_window_backend = None

        try:
            import matplotlib.pyplot as plt

            if self._debug_window_backend is None:
                plt.ion()
                self._debug_figure, self._debug_axes = plt.subplots(
                    num="vision training progress"
                )
                self._debug_axes.axis("off")
                self._debug_image_artist = self._debug_axes.imshow(panel)
                self._debug_window_backend = "matplotlib"
            elif self._debug_image_artist is not None:
                self._debug_image_artist.set_data(panel)
            if self._debug_figure is not None:
                self._debug_figure.canvas.draw_idle()
                self._debug_figure.canvas.flush_events()
                plt.pause(0.001)
        except Exception as error:
            self._debug_window_failed = True
            self.get_logger().warning(
                "Could not open a local training visualization window with OpenCV "
                f"or matplotlib: {error}"
            )

    def _write_metrics(self, metrics: dict[str, float]) -> None:
        self._writer.add_scalar("train/total_loss", metrics["loss"], self._train_step)
        self._writer.add_scalar("train/value_loss", metrics["value_loss"], self._train_step)
        self._writer.add_scalar(
            "train/value_accuracy",
            metrics["value_accuracy"],
            self._train_step,
        )
        if metrics["identity_loss"] > 0.0:
            self._writer.add_scalar(
                "train/identity_loss",
                metrics["identity_loss"],
                self._train_step,
            )
            self._writer.add_scalar("train/depth_loss", metrics["depth_loss"], self._train_step)
            self._writer.add_scalar(
                "train/missed_occupancy_loss",
                metrics["missed_occupancy_loss"],
                self._train_step,
            )
        self._writer.add_scalar("train/grad_norm", metrics["grad_norm"], self._train_step)
        self._writer.add_scalar(
            "data/train_feedback",
            len(self._feedback_buffer),
            self._train_step,
        )
        self._writer.add_scalar(
            "data/test_feedback",
            len(self._test_feedback_buffer),
            self._train_step,
        )
        self._writer.add_scalar(
            "data/train_samples",
            len(self._train_sample_ids),
            self._train_step,
        )
        self._writer.add_scalar(
            "data/test_samples",
            len(self._test_sample_ids),
            self._train_step,
        )

    def _write_eval_metrics(self, metrics: dict[str, float], batch_size: int) -> None:
        self._writer.add_scalar("eval/total_loss", metrics["loss"], self._train_step)
        self._writer.add_scalar("eval/value_loss", metrics["value_loss"], self._train_step)
        self._writer.add_scalar(
            "eval/value_accuracy",
            metrics["value_accuracy"],
            self._train_step,
        )
        if metrics["identity_loss"] > 0.0:
            self._writer.add_scalar(
                "eval/identity_loss",
                metrics["identity_loss"],
                self._train_step,
            )
            self._writer.add_scalar("eval/depth_loss", metrics["depth_loss"], self._train_step)
            self._writer.add_scalar(
                "eval/missed_occupancy_loss",
                metrics["missed_occupancy_loss"],
                self._train_step,
            )
        self._writer.add_scalar(
            "data/eval_feedback_batch",
            min(self._eval_batch_size, len(self._test_feedback_buffer)),
            self._train_step,
        )
        if batch_size:
            self._writer.add_scalar("data/eval_sample_batch", batch_size, self._train_step)
        self._writer.add_scalar(
            "data/test_feedback",
            len(self._test_feedback_buffer),
            self._train_step,
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "tensorboard_base_log_dir": str(self._tensorboard_log_dir),
            "tensorboard_run_name": self._tensorboard_run_name,
            "tensorboard_log_dir": str(self._tensorboard_run_dir),
            "optimizer": type(self._optimizer).__name__,
            "learning_rate": self._learning_rate,
            "replay_dir": str(self._replay_dir),
            "sample_buffer_capacity": self._sample_buffer_capacity,
            "sample_prefetch_size": self._sample_prefetch_size,
            "sample_loader_workers": self._sample_loader_workers,
            "min_sample_size": self._min_sample_size,
            "train_batch_size": self._train_batch_size,
            "train_split_threshold": self._train_split_threshold,
            "eval_interval_steps": self._eval_interval_steps,
            "eval_batch_size": self._eval_batch_size,
            "train_steps_per_tick": self._train_steps_per_tick,
            "identity_loss_weight": self._identity_loss_weight,
            "missed_occupancy_loss_weight": self._missed_occupancy_loss_weight,
            "depth_loss_weight": self._depth_loss_weight,
            "value_loss_weight": self._value_loss_weight,
            "checkpoint_save_interval": self._checkpoint_save_interval,
        }

    def _save_checkpoint(self) -> None:
        save_checkpoint(
            self._latest_checkpoint_path,
            self._model,
            self._optimizer,
            self._scaler,
            epoch=0,
            global_step=self._train_step,
            config=self._checkpoint_config,
            metadata=self._checkpoint_metadata(),
        )
        self.get_logger().info(
            f"Saved online vision checkpoint path={self._latest_checkpoint_path}; "
            f"train_step={self._train_step}; "
            f"tensorboard_log_dir={self._tensorboard_run_dir}"
        )

    def destroy_node(self) -> bool:
        self._save_checkpoint()
        self._writer.flush()
        self._writer.close()
        if self._sample_executor is not None:
            self._sample_executor.shutdown(wait=False, cancel_futures=True)
        if self._debug_window_backend == "opencv":
            try:
                cv2.destroyWindow("vision training progress")
            except cv2.error:
                pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = VisionOnlineTrainerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
