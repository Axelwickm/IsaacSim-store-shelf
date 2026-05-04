#!/usr/bin/env python3

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from rclpy.node import Node
from torch.utils.data import DataLoader

from vision.checkpoints import load_checkpoint, save_checkpoint
from vision.dataset import (
    DEFAULT_IMAGE_SIZE,
    StoreShelfVisionDataset,
    collate_vision_samples,
)
from vision.model import (
    HIDDEN_DIM,
    IDENTITY_ALPHA_EPSILON,
    LATENT_DIM,
    NUM_QUERIES,
    PATCH_SIZE,
    create_query_model,
    model_device,
)


DEFAULT_DATASET_DIR = Path("/workspace/collect_vision_data_output")
DEFAULT_CHECKPOINT_DIR = Path("/workspace/checkpoints/vision")
DEFAULT_BATCH_SIZE = 8
DEFAULT_SPLIT = "train"
DEFAULT_MODE = "train"
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_LOG_EVERY = 10
DEFAULT_MAX_EPOCHS = 1000
DEFAULT_SAVE_EVERY_EPOCHS = 1
DISPLAY_SCALE = 0.3
DEFAULT_USE_MIXED_PRECISION = True
DISPLAY_INTERVAL_SECONDS = 5.0
DEPTH_DISPLAY_MAX_METERS = 2.0
DEPTH_LOSS_WEIGHT = 0.1
IDENTITY_LOSS_WEIGHT = 1.0
MISSED_OCCUPANCY_PENALTY_WEIGHT = 1.0
DEBUG_RENDER_STATS = True


def _to_display_rgb(rgb_tensor) -> np.ndarray:
    rgb = rgb_tensor.detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _to_display_identity(identity_tensor) -> np.ndarray:
    identity = np.asarray(identity_tensor).astype(np.uint32)
    red = ((identity * 67) % 251).astype(np.uint8)
    green = ((identity * 29 + 71) % 253).astype(np.uint8)
    blue = ((identity * 53 + 149) % 255).astype(np.uint8)
    background = identity == 0
    red[background] = 0
    green[background] = 0
    blue[background] = 0
    return np.stack([blue, green, red], axis=-1)


def _to_display_depth(depth_tensor) -> np.ndarray:
    depth = np.asarray(depth_tensor).astype(np.float32)
    while depth.ndim > 2:
        depth = depth.squeeze(0)
    valid_mask = depth > 0.0
    normalized = np.clip(depth / DEPTH_DISPLAY_MAX_METERS, 0.0, 1.0)
    normalized = (normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    colored[~valid_mask] = 0
    return colored


def _to_display_occupancy(occupancy_tensor) -> np.ndarray:
    occupancy = np.asarray(occupancy_tensor).astype(np.float32)
    while occupancy.ndim > 2:
        occupancy = occupancy.squeeze(0)
    occupancy = np.clip(occupancy, 0.0, 1.0)
    grayscale = (occupancy * 255.0).astype(np.uint8)
    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


def _log_render_debug_stats(
    node: Node,
    mode: str,
    epoch_index: int,
    global_step: int,
    outputs: dict,
    alpha_targets: torch.Tensor | None,
) -> None:
    patch_alpha_logits = outputs["patch_alpha_logits"].detach().float()
    patch_alpha = outputs["patch_alpha"].detach().float()
    predicted_occupancy = outputs["predicted_occupancy"].detach().float()
    predicted_depth = outputs["predicted_depth"].detach().float()
    sizes = outputs["sizes"].detach().float()
    depths = outputs["depth"].detach().float()

    occupancy_fill_fraction = (
        (predicted_occupancy > 0.05).float().mean(dim=(-3, -2, -1))
    )
    log_message = (
        f"[debug-render] mode={mode} epoch={epoch_index} step={global_step} "
        f"logits_mean={patch_alpha_logits.mean().item():.4f} "
        f"logits_std={patch_alpha_logits.std().item():.4f} "
        f"alpha_mean={patch_alpha.mean().item():.4f} "
        f"alpha_std={patch_alpha.std().item():.4f} "
        f"alpha_min={patch_alpha.min().item():.4f} "
        f"alpha_max={patch_alpha.max().item():.4f} "
        f"size_mean=({sizes[..., 0].mean().item():.4f},{sizes[..., 1].mean().item():.4f}) "
        f"depth_mean={depths.mean().item():.4f} "
        f"occupancy_fill_mean={occupancy_fill_fraction.mean().item():.4f} "
        f"occupancy_max={predicted_occupancy.max().item():.4f} "
        f"depth_render_max={predicted_depth.max().item():.4f}"
    )
    node.get_logger().info(log_message)
    if alpha_targets is not None:
        target_fill_fraction = (
            (alpha_targets.detach().float() > 0.5).float().mean(dim=(-2, -1))
        )
        node.get_logger().info(
            f"[debug-render] alpha_target_fill_mean={target_fill_fraction.mean().item():.4f} "
            f"alpha_target_fill_max={target_fill_fraction.max().item():.4f}"
        )


def _scale_for_display(image: np.ndarray) -> np.ndarray:
    return cv2.resize(
        image,
        dsize=None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA,
    )


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.putText(
        labeled,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return cv2.resize(
        image,
        (reference.shape[1], reference.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def _build_alpha_targets(
    instance_targets: torch.Tensor,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    depths: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_queries = centers.shape[:2]
    image_height = instance_targets.shape[-2]
    image_width = instance_targets.shape[-1]
    device = instance_targets.device

    center_x = ((centers[..., 0] + 1.0) * 0.5 * (image_width - 1)).round().long()
    center_y = ((centers[..., 1] + 1.0) * 0.5 * (image_height - 1)).round().long()
    center_x = center_x.clamp(0, image_width - 1)
    center_y = center_y.clamp(0, image_height - 1)
    batch_index = torch.arange(batch_size, device=device)[:, None].expand(
        -1, num_queries
    )
    center_instance_ids = instance_targets[batch_index, center_y, center_x]
    center_instance_ids = _unique_center_instance_ids(center_instance_ids, depths)

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
        instance_targets[:, None, :, :] == center_instance_ids[:, :, None, None]
    ).float()
    sampled_masks = torch.nn.functional.grid_sample(
        target_masks.view(batch_size * num_queries, 1, image_height, image_width),
        sample_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch_size, num_queries, patch_size, patch_size)

    valid_instance = (
        (center_instance_ids > 0).float().view(batch_size, num_queries, 1, 1)
    )
    return sampled_masks, valid_instance


def _center_instance_ids(
    instance_targets: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_queries = centers.shape[:2]
    image_height = instance_targets.shape[-2]
    image_width = instance_targets.shape[-1]
    center_x = ((centers[..., 0] + 1.0) * 0.5 * (image_width - 1)).round().long()
    center_y = ((centers[..., 1] + 1.0) * 0.5 * (image_height - 1)).round().long()
    center_x = center_x.clamp(0, image_width - 1)
    center_y = center_y.clamp(0, image_height - 1)
    batch_index = torch.arange(batch_size, device=instance_targets.device)[
        :, None
    ].expand(-1, num_queries)
    return instance_targets[batch_index, center_y, center_x]


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


def _presence_targets(
    instance_targets: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    center_ids = _center_instance_ids(instance_targets, centers)
    return (center_ids > 0).float()


def _render_assigned_identity_for_display(
    patch_alpha: torch.Tensor,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    depths: torch.Tensor,
    center_ids: torch.Tensor,
    image_height: int,
    image_width: int,
) -> np.ndarray:
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
    local_x = ((grid_x - centers_x) / sizes_x).clamp(-1.1, 1.1)
    local_y = ((grid_y - centers_y) / sizes_y).clamp(-1.1, 1.1)
    sample_grid = torch.stack([local_x, local_y], dim=-1)
    sampled_alpha = torch.nn.functional.grid_sample(
        patch_alpha.unsqueeze(1),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)

    query_order = torch.argsort(depths.flatten(), descending=False)
    rendered_identity = torch.zeros(
        image_height, image_width, device=device, dtype=torch.int64
    )
    remaining_identity = torch.ones(
        image_height, image_width, device=device, dtype=sampled_alpha.dtype
    )
    for query_index in query_order.tolist():
        center_id = int(center_ids[query_index].item())
        if center_id <= 0:
            continue
        alpha = sampled_alpha[query_index]
        claim_strength = alpha * remaining_identity
        rendered_identity = torch.where(
            claim_strength > IDENTITY_ALPHA_EPSILON,
            torch.full_like(rendered_identity, center_id),
            rendered_identity,
        )
        remaining_identity = remaining_identity * (1.0 - alpha)

    return rendered_identity.cpu().numpy()


def main() -> None:
    rclpy.init()
    node = Node("vision_trainer")
    node.declare_parameter("dataset_dir", str(DEFAULT_DATASET_DIR))
    node.declare_parameter("checkpoint_dir", str(DEFAULT_CHECKPOINT_DIR))
    node.declare_parameter("checkpoint_path", "")
    node.declare_parameter("batch_size", DEFAULT_BATCH_SIZE)
    node.declare_parameter("split", DEFAULT_SPLIT)
    node.declare_parameter("mode", DEFAULT_MODE)
    node.declare_parameter("learning_rate", DEFAULT_LEARNING_RATE)
    node.declare_parameter("log_every", DEFAULT_LOG_EVERY)
    node.declare_parameter("max_epochs", DEFAULT_MAX_EPOCHS)
    node.declare_parameter("resume", True)
    node.declare_parameter("save_every_epochs", DEFAULT_SAVE_EVERY_EPOCHS)
    node.declare_parameter("image_size", DEFAULT_IMAGE_SIZE)
    node.declare_parameter("use_mixed_precision", DEFAULT_USE_MIXED_PRECISION)
    dataset_dir = Path(str(node.get_parameter("dataset_dir").value)).resolve()
    checkpoint_dir = Path(str(node.get_parameter("checkpoint_dir").value)).resolve()
    checkpoint_path_value = str(node.get_parameter("checkpoint_path").value).strip()
    batch_size = int(node.get_parameter("batch_size").value)
    split = str(node.get_parameter("split").value).strip()
    mode = str(node.get_parameter("mode").value).strip()
    learning_rate = float(node.get_parameter("learning_rate").value)
    log_every = int(node.get_parameter("log_every").value)
    max_epochs = int(node.get_parameter("max_epochs").value)
    resume = bool(node.get_parameter("resume").value)
    save_every_epochs = int(node.get_parameter("save_every_epochs").value)
    image_size = int(node.get_parameter("image_size").value)
    use_mixed_precision = bool(node.get_parameter("use_mixed_precision").value)

    if mode not in {"train", "eval", "inference"}:
        raise ValueError(
            f"Unsupported mode {mode!r}; expected 'train', 'eval', or 'inference'"
        )

    dataset = StoreShelfVisionDataset(
        dataset_dir,
        split=split,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_vision_samples,
    )
    device = model_device()
    model = create_query_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_mixed_precision and device.type == "cuda"
    )
    last_display_time = 0.0
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    checkpoint_path = (
        Path(checkpoint_path_value).resolve()
        if checkpoint_path_value
        else latest_checkpoint_path
    )
    checkpoint_config = {
        "num_queries": NUM_QUERIES,
        "latent_dim": LATENT_DIM,
        "patch_size": PATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "value_in_slot_head": True,
    }
    start_epoch = 0
    global_step = 0
    logged_render_debug = False

    should_load_checkpoint = (mode == "train" and resume) or mode in {
        "eval",
        "inference",
    }
    if should_load_checkpoint and checkpoint_path.exists():
        checkpoint_metadata = load_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer if mode == "train" else None,
            scaler=scaler if mode == "train" else None,
            expected_config=checkpoint_config,
            map_location=device,
            strict=False,
        )
        if mode == "train":
            start_epoch = checkpoint_metadata["epoch"] + 1
            global_step = checkpoint_metadata["global_step"]
        node.get_logger().info(
            f"Loaded checkpoint from {checkpoint_path} at "
            f"epoch={checkpoint_metadata['epoch']} "
            f"global_step={checkpoint_metadata['global_step']}"
        )
    elif mode in {"eval", "inference"}:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    node.get_logger().info(
        f"Looping through {len(dataset)} {split} samples from {dataset_dir} "
        f"with batch_size={batch_size} in mode={mode} on device={device} "
        f"image_size={image_size} mixed_precision={use_mixed_precision}"
    )
    node.get_logger().info("Press q in the preview window to stop")

    try:
        for epoch_index in range(start_epoch, max_epochs):
            if not rclpy.ok():
                break
            epoch_start_time = time.perf_counter()
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            for batch in loader:
                pixel_values = batch["rgb_normalized"].to(device)
                identity_targets = batch["instance_segmentation"].to(device)
                depth_targets = batch["distance_to_camera"].to(device)
                gt_foreground = (identity_targets > 0).float().unsqueeze(1)

                if mode == "train":
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        enabled=use_mixed_precision and device.type == "cuda",
                    ):
                        outputs = model(pixel_values)
                        predicted_depth = outputs["predicted_depth"]
                        alpha_targets, alpha_target_weights = _build_alpha_targets(
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
                    with torch.autocast(device_type=device.type, enabled=False):
                        predicted_occupancy = outputs["predicted_occupancy"].float()
                        depth_error = F.smooth_l1_loss(
                            predicted_depth.float(),
                            depth_targets.float(),
                            reduction="none",
                        )
                        depth_loss = (
                            depth_error * predicted_occupancy
                        ).sum() / predicted_occupancy.sum().clamp_min(1e-6)
                        missed_occupancy_penalty = (
                            gt_foreground.float() * (1.0 - predicted_occupancy)
                        ).mean()
                        extra_occupancy_penalty = (
                            (1.0 - gt_foreground.float()) * predicted_occupancy
                        ).mean()
                        loss = (
                            DEPTH_LOSS_WEIGHT * depth_loss
                            + IDENTITY_LOSS_WEIGHT * identity_loss
                            + MISSED_OCCUPANCY_PENALTY_WEIGHT * missed_occupancy_penalty
                        )
                    if DEBUG_RENDER_STATS and not logged_render_debug:
                        _log_render_debug_stats(
                            node,
                            mode,
                            epoch_index,
                            global_step,
                            outputs,
                            alpha_targets,
                        )
                        logged_render_debug = True
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                elif mode == "eval":
                    model.eval()
                    with torch.no_grad():
                        with torch.autocast(
                            device_type=device.type,
                            enabled=use_mixed_precision and device.type == "cuda",
                        ):
                            outputs = model(pixel_values)
                            predicted_depth = outputs["predicted_depth"]
                            alpha_targets, alpha_target_weights = _build_alpha_targets(
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
                            identity_loss = (
                                identity_loss
                                / alpha_target_weights.expand_as(alpha_targets)
                                .sum()
                                .clamp_min(1.0)
                            )
                        with torch.autocast(device_type=device.type, enabled=False):
                            predicted_occupancy = outputs["predicted_occupancy"].float()
                            depth_error = F.smooth_l1_loss(
                                predicted_depth.float(),
                                depth_targets.float(),
                                reduction="none",
                            )
                            depth_loss = (
                                depth_error * predicted_occupancy
                            ).sum() / predicted_occupancy.sum().clamp_min(1e-6)
                            missed_occupancy_penalty = (
                                gt_foreground.float() * (1.0 - predicted_occupancy)
                            ).mean()
                            extra_occupancy_penalty = (
                                (1.0 - gt_foreground.float()) * predicted_occupancy
                            ).mean()
                            loss = (
                                DEPTH_LOSS_WEIGHT * depth_loss
                                + IDENTITY_LOSS_WEIGHT * identity_loss
                                + MISSED_OCCUPANCY_PENALTY_WEIGHT
                                * missed_occupancy_penalty
                                * extra_occupancy_penalty
                            )
                        if DEBUG_RENDER_STATS and not logged_render_debug:
                            _log_render_debug_stats(
                                node,
                                mode,
                                epoch_index,
                                global_step,
                                outputs,
                                alpha_targets,
                            )
                            logged_render_debug = True
                else:
                    model.eval()
                    with torch.no_grad():
                        with torch.autocast(
                            device_type=device.type,
                            enabled=use_mixed_precision and device.type == "cuda",
                        ):
                            outputs = model(pixel_values)
                        if DEBUG_RENDER_STATS and not logged_render_debug:
                            _log_render_debug_stats(
                                node,
                                mode,
                                epoch_index,
                                global_step,
                                outputs,
                                alpha_targets=None,
                            )
                            logged_render_debug = True
                        loss = outputs["predicted_depth"].sum() * 0.0

                batch_loss = float(loss.detach().cpu())
                batch_size_actual = len(batch["sample_id"])
                global_step += 1
                epoch_loss_sum += batch_loss * batch_size_actual
                epoch_sample_count += batch_size_actual
                elapsed = max(time.perf_counter() - epoch_start_time, 1e-6)
                samples_per_second = epoch_sample_count / elapsed
                average_loss = epoch_loss_sum / max(epoch_sample_count, 1)

                if global_step == 1 or global_step % max(log_every, 1) == 0:
                    node.get_logger().info(
                        f"[{mode}] epoch={epoch_index} step={global_step} "
                        f"batch_loss={batch_loss:.4f} avg_loss={average_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"samples_per_sec={samples_per_second:.2f}"
                    )

                now = time.perf_counter()
                if now - last_display_time >= DISPLAY_INTERVAL_SECONDS:
                    last_display_time = now
                    center_instance_ids = _center_instance_ids(
                        identity_targets,
                        outputs["centers"].detach(),
                    )
                    center_instance_ids = _unique_center_instance_ids(
                        center_instance_ids,
                        outputs["depth"].detach(),
                    )
                    for batch_index, sample_id in enumerate(batch["sample_id"]):
                        rgb = _to_display_rgb(batch["rgb"][batch_index])
                        gt_identity = _to_display_identity(
                            batch["instance_segmentation"][batch_index]
                        )
                        depth = _to_display_depth(
                            batch["distance_to_camera"][batch_index]
                        )
                        gt_occupancy = _to_display_occupancy(
                            gt_foreground[batch_index].detach().cpu().numpy()
                        )
                        center_ids = center_instance_ids[batch_index].detach()
                        remapped_identity = _render_assigned_identity_for_display(
                            outputs["patch_alpha"][batch_index].detach(),
                            outputs["centers"][batch_index].detach(),
                            outputs["sizes"][batch_index].detach(),
                            outputs["depth"][batch_index].detach(),
                            center_ids,
                            image_height=rgb.shape[0],
                            image_width=rgb.shape[1],
                        )
                        predicted_identity = _to_display_identity(remapped_identity)
                        predicted_depth = (
                            outputs["predicted_depth"][batch_index]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                        predicted_occupancy = _to_display_occupancy(
                            outputs["predicted_occupancy"][batch_index]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                        gt_identity = _resize_like(gt_identity, rgb)
                        predicted_identity = _resize_like(predicted_identity, rgb)
                        predicted_depth = _to_display_depth(predicted_depth)
                        predicted_depth = _resize_like(predicted_depth, rgb)
                        gt_occupancy = _resize_like(gt_occupancy, rgb)
                        predicted_occupancy = _resize_like(predicted_occupancy, rgb)
                        rgb = _label_panel(_scale_for_display(rgb), "rgb")
                        gt_identity = _label_panel(
                            _scale_for_display(gt_identity),
                            "gt identity",
                        )
                        depth = _label_panel(_scale_for_display(depth), "gt depth")
                        predicted_identity = _label_panel(
                            _scale_for_display(predicted_identity),
                            "pred identity",
                        )
                        predicted_depth = _label_panel(
                            _scale_for_display(predicted_depth),
                            "pred depth",
                        )
                        top_row = np.hstack([rgb, gt_identity, depth])
                        bottom_row = np.hstack(
                            [
                                rgb.copy(),
                                predicted_identity,
                                predicted_depth,
                            ]
                        )
                        preview = np.vstack([top_row, bottom_row])
                        cv2.imshow("vision_train_loop", preview)
                        key = cv2.waitKey(1) & 255
                        if key in (ord("q"), 27):
                            raise KeyboardInterrupt
            epoch_elapsed = max(time.perf_counter() - epoch_start_time, 1e-6)
            epoch_average_loss = epoch_loss_sum / max(epoch_sample_count, 1)
            node.get_logger().info(
                f"[{mode}] epoch={epoch_index} summary "
                f"samples={epoch_sample_count} avg_loss={epoch_average_loss:.4f} "
                f"elapsed_sec={epoch_elapsed:.2f} "
                f"samples_per_sec={epoch_sample_count / epoch_elapsed:.2f}"
            )
            if mode == "train":
                save_checkpoint(
                    latest_checkpoint_path,
                    model,
                    optimizer,
                    scaler,
                    epoch_index,
                    global_step,
                    checkpoint_config,
                )
                if (epoch_index + 1) % max(save_every_epochs, 1) == 0:
                    epoch_checkpoint_path = (
                        checkpoint_dir / f"epoch_{epoch_index + 1:06d}.pt"
                    )
                    save_checkpoint(
                        epoch_checkpoint_path,
                        model,
                        optimizer,
                        scaler,
                        epoch_index,
                        global_step,
                        checkpoint_config,
                    )
    except KeyboardInterrupt:
        node.get_logger().info("Stopping vision dataset loop")
    finally:
        cv2.destroyAllWindows()

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
