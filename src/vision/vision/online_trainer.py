#!/usr/bin/env python3

import json
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
import torch
import torch.nn.functional as F

from vision.checkpoints import load_checkpoint, save_checkpoint
from vision.dataset import RGB_MEAN, RGB_STD
from vision.inference import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_SAVE_INTERVAL,
    DEFAULT_DEPTH_LOSS_WEIGHT,
    DEFAULT_GEOMETRY_LOSS_WEIGHT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_REPLAY_SIZE,
    DEFAULT_PRESENCE_LOSS_WEIGHT,
    DEFAULT_REPLAY_BUFFER_CAPACITY,
    DEFAULT_REPLAY_DIR,
    DEFAULT_TENSORBOARD_DIR,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_STEPS_PER_TICK,
    DEFAULT_TRAIN_TICK_PERIOD,
    _parameter_bool,
)
from vision.model import HIDDEN_DIM, LATENT_DIM, NUM_QUERIES, create_query_model, model_device

try:
    from torch.utils.tensorboard import SummaryWriter
    _SUMMARY_WRITER_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None
    _SUMMARY_WRITER_IMPORT_ERROR = error


def _default_run_name() -> str:
    return "vision-online-" + datetime.now().strftime("%Y%m%d-%H%M%S")


class VisionOnlineTrainerNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_online_trainer")
        self.declare_parameter("checkpoint_dir", str(DEFAULT_CHECKPOINT_DIR))
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("replay_dir", str(DEFAULT_REPLAY_DIR))
        self.declare_parameter("image_size", 512)
        self.declare_parameter("use_mixed_precision", True)
        self.declare_parameter("replay_buffer_capacity", DEFAULT_REPLAY_BUFFER_CAPACITY)
        self.declare_parameter("min_replay_size", DEFAULT_MIN_REPLAY_SIZE)
        self.declare_parameter("train_batch_size", DEFAULT_TRAIN_BATCH_SIZE)
        self.declare_parameter("train_steps_per_tick", DEFAULT_TRAIN_STEPS_PER_TICK)
        self.declare_parameter("train_tick_period_sec", DEFAULT_TRAIN_TICK_PERIOD)
        self.declare_parameter("online_learning_rate", DEFAULT_LEARNING_RATE)
        self.declare_parameter("geometry_loss_weight", DEFAULT_GEOMETRY_LOSS_WEIGHT)
        self.declare_parameter("presence_loss_weight", DEFAULT_PRESENCE_LOSS_WEIGHT)
        self.declare_parameter("depth_loss_weight", DEFAULT_DEPTH_LOSS_WEIGHT)
        self.declare_parameter("freeze_backbone", True)
        self.declare_parameter("tensorboard_log_dir", str(DEFAULT_TENSORBOARD_DIR))
        self.declare_parameter("tensorboard_run_name", "")
        self.declare_parameter("checkpoint_save_interval", DEFAULT_CHECKPOINT_SAVE_INTERVAL)
        self.declare_parameter("replay_scan_period_sec", 1.0)

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
        self._replay_buffer_capacity = max(
            int(self.get_parameter("replay_buffer_capacity").value),
            1,
        )
        self._min_replay_size = max(int(self.get_parameter("min_replay_size").value), 1)
        self._train_batch_size = max(int(self.get_parameter("train_batch_size").value), 1)
        self._train_steps_per_tick = max(
            int(self.get_parameter("train_steps_per_tick").value),
            1,
        )
        self._train_tick_period_sec = max(
            float(self.get_parameter("train_tick_period_sec").value),
            0.001,
        )
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
        ).strip() or _default_run_name()
        self._tensorboard_run_dir = self._tensorboard_log_dir / self._tensorboard_run_name
        self._checkpoint_save_interval = max(
            int(self.get_parameter("checkpoint_save_interval").value),
            1,
        )
        replay_scan_period_sec = max(
            float(self.get_parameter("replay_scan_period_sec").value),
            0.1,
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
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "value_in_slot_head": True,
        }
        if self._freeze_backbone:
            for name, parameter in self._model.named_parameters():
                parameter.requires_grad = (
                    name.startswith("query_embed")
                    or name.startswith("decoder")
                    or name.startswith("slot_head")
                )
        trainable_parameters = [
            parameter for parameter in self._model.parameters() if parameter.requires_grad
        ]
        self._optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self._online_learning_rate,
        )
        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self._use_mixed_precision and self._device.type == "cuda",
        )
        checkpoint_metadata = {"global_step": 0}
        if checkpoint_path.exists():
            checkpoint_metadata = load_checkpoint(
                checkpoint_path,
                self._model,
                optimizer=self._optimizer,
                scaler=self._scaler,
                expected_config=self._checkpoint_config,
                map_location=self._device,
                strict=False,
            )
        self._model.train()
        self._train_step = int(checkpoint_metadata["global_step"])
        self._replay_buffer: deque[dict[str, Any]] = deque(maxlen=self._replay_buffer_capacity)
        self._loaded_replay_paths: set[str] = set()
        self._writer = SummaryWriter(log_dir=str(self._tensorboard_run_dir))
        self._last_wait_log_size = -1

        self._scan_timer = self.create_timer(replay_scan_period_sec, self._scan_replay_dir)
        self._train_timer = self.create_timer(
            self._train_tick_period_sec,
            self._handle_training_tick,
        )
        self._scan_replay_dir()

        self.get_logger().info(
            "Vision online trainer ready "
            f"(replay_dir={self._replay_dir}, checkpoint={self._latest_checkpoint_path}, "
            f"device={self._device}, replay_capacity={self._replay_buffer_capacity}, "
            f"batch_size={self._train_batch_size}, steps_per_tick={self._train_steps_per_tick})"
        )

    def _normalize_rgb_batch(self, rgb_batch: np.ndarray) -> torch.Tensor:
        pixel_values = torch.from_numpy(rgb_batch).permute(0, 3, 1, 2).float() / 255.0
        normalized = (pixel_values - RGB_MEAN.view(1, 3, 1, 1)) / RGB_STD.view(1, 3, 1, 1)
        return normalized.to(self._device)

    def _scan_replay_dir(self) -> None:
        if not self._replay_dir.exists():
            return
        loaded_count = 0
        for path in sorted(self._replay_dir.glob("*.npz")):
            path_key = str(path)
            if path_key in self._loaded_replay_paths:
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata_json"]))
                    rgb = np.ascontiguousarray(data["rgb"])
            except Exception as error:
                self.get_logger().warning(f"Skipping invalid replay sample {path}: {error}")
                continue
            metadata["rgb"] = rgb
            self._replay_buffer.append(metadata)
            self._loaded_replay_paths.add(path_key)
            loaded_count += 1
        if loaded_count:
            self.get_logger().info(
                f"Loaded {loaded_count} replay samples; "
                f"replay_size={len(self._replay_buffer)}/{self._replay_buffer_capacity}"
            )

    def _handle_training_tick(self) -> None:
        if len(self._replay_buffer) < self._min_replay_size:
            if len(self._replay_buffer) != self._last_wait_log_size:
                self._last_wait_log_size = len(self._replay_buffer)
                self.get_logger().info(
                    "Online trainer waiting for replay buffer: "
                    f"replay_size={len(self._replay_buffer)}/{self._min_replay_size}"
                )
            return
        for _ in range(self._train_steps_per_tick):
            metrics = self._run_one_train_step()
        self._writer.flush()
        if self._train_step % max(self._checkpoint_save_interval, 1) < self._train_steps_per_tick:
            self.get_logger().info(
                "Online trainer step: "
                f"train_step={self._train_step}; "
                f"replay_size={len(self._replay_buffer)}; "
                f"loss={metrics['loss']:.6f}; "
                f"accuracy={metrics['accuracy']:.3f}"
            )

    def _run_one_train_step(self) -> dict[str, float]:
        batch = random.sample(
            list(self._replay_buffer),
            k=min(self._train_batch_size, len(self._replay_buffer)),
        )
        rgb_batch = np.stack([sample["rgb"] for sample in batch], axis=0)
        pixel_values = self._normalize_rgb_batch(rgb_batch)
        query_indices = torch.tensor(
            [sample["query_index"] for sample in batch],
            device=self._device,
            dtype=torch.long,
        )
        arm_indices = torch.tensor(
            [sample["arm_index"] for sample in batch],
            device=self._device,
            dtype=torch.long,
        )
        labels = torch.tensor(
            [sample["label"] for sample in batch],
            device=self._device,
            dtype=torch.float32,
        )

        self._optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self._device.type,
            enabled=self._use_mixed_precision and self._device.type == "cuda",
        ):
            outputs = self._model(pixel_values)
            batch_indices = torch.arange(len(batch), device=self._device)
            selected_logits = outputs["value_logits"][batch_indices, query_indices, arm_indices]
            value_loss = F.binary_cross_entropy_with_logits(selected_logits, labels)
            center_loss_terms = []
            depth_loss_terms = []
            presence_loss_terms = []
            for sample_index, sample in enumerate(batch):
                gt_targets = list(sample.get("gt_targets") or [])[:NUM_QUERIES]
                if not gt_targets:
                    continue
                target_centers = torch.tensor(
                    [[target["center_x"], target["center_y"]] for target in gt_targets],
                    device=self._device,
                    dtype=outputs["centers"].dtype,
                )
                target_depth = torch.tensor(
                    [[target["depth_m"]] for target in gt_targets],
                    device=self._device,
                    dtype=outputs["depth"].dtype,
                )
                target_presence = torch.zeros(
                    NUM_QUERIES,
                    device=self._device,
                    dtype=outputs["presence_logits"].dtype,
                )
                target_presence[: len(gt_targets)] = 1.0
                center_loss_terms.append(
                    F.smooth_l1_loss(
                        outputs["centers"][sample_index, : len(gt_targets)],
                        target_centers,
                    )
                )
                depth_loss_terms.append(
                    F.smooth_l1_loss(
                        outputs["depth"][sample_index, : len(gt_targets)],
                        target_depth,
                    )
                )
                presence_loss_terms.append(
                    F.binary_cross_entropy_with_logits(
                        outputs["presence_logits"][sample_index, :, 0],
                        target_presence,
                    )
                )
            zero_loss = value_loss * 0.0
            center_loss = torch.stack(center_loss_terms).mean() if center_loss_terms else zero_loss
            depth_loss = torch.stack(depth_loss_terms).mean() if depth_loss_terms else zero_loss
            presence_loss = (
                torch.stack(presence_loss_terms).mean() if presence_loss_terms else zero_loss
            )
            loss = (
                value_loss
                if not center_loss_terms
                else value_loss
                + self._geometry_loss_weight * center_loss
                + self._depth_loss_weight * depth_loss
                + self._presence_loss_weight * presence_loss
            )

        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._train_step += 1
        with torch.no_grad():
            probabilities = torch.sigmoid(selected_logits.detach())
            accuracy = ((probabilities >= 0.5) == (labels >= 0.5)).float().mean()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "accuracy": float(accuracy.cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "center_loss": float(center_loss.detach().cpu()),
            "depth_loss": float(depth_loss.detach().cpu()),
            "presence_loss": float(presence_loss.detach().cpu()),
        }
        self._write_metrics(metrics)
        if self._train_step % self._checkpoint_save_interval == 0:
            self._save_checkpoint()
        return metrics

    def _write_metrics(self, metrics: dict[str, float]) -> None:
        self._writer.add_scalar("train/total_loss", metrics["loss"], self._train_step)
        self._writer.add_scalar("train/value_loss", metrics["value_loss"], self._train_step)
        self._writer.add_scalar("train/gt_center_loss", metrics["center_loss"], self._train_step)
        self._writer.add_scalar("train/gt_depth_loss", metrics["depth_loss"], self._train_step)
        self._writer.add_scalar("train/gt_presence_loss", metrics["presence_loss"], self._train_step)
        self._writer.add_scalar("train/value_accuracy", metrics["accuracy"], self._train_step)
        self._writer.add_scalar("replay/buffer_size", len(self._replay_buffer), self._train_step)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "tensorboard_base_log_dir": str(self._tensorboard_log_dir),
            "tensorboard_run_name": self._tensorboard_run_name,
            "tensorboard_log_dir": str(self._tensorboard_run_dir),
            "online_training_enabled": True,
            "optimizer": type(self._optimizer).__name__,
            "learning_rate": self._online_learning_rate,
            "replay_buffer_capacity": self._replay_buffer_capacity,
            "min_replay_size": self._min_replay_size,
            "train_batch_size": self._train_batch_size,
            "train_steps_per_tick": self._train_steps_per_tick,
            "geometry_loss_weight": self._geometry_loss_weight,
            "presence_loss_weight": self._presence_loss_weight,
            "depth_loss_weight": self._depth_loss_weight,
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
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = VisionOnlineTrainerNode()
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
