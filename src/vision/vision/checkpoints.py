from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    epoch: int,
    global_step: int,
    config: dict,
    metadata: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    if metadata is not None:
        checkpoint["metadata"] = metadata
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    expected_config: dict | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    if expected_config is not None:
        for key, expected_value in expected_config.items():
            checkpoint_value = checkpoint_config.get(key)
            if checkpoint_value is None:
                continue
            if checkpoint_value != expected_value:
                raise ValueError(
                    f"Checkpoint config mismatch for {key}: "
                    f"{checkpoint_value!r} != {expected_value!r}"
                )

    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )
    optimizer_loaded = optimizer is not None and "optimizer_state_dict" in checkpoint
    scaler_loaded = scaler is not None and "scaler_state_dict" in checkpoint
    if optimizer_loaded:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler_loaded:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "config": checkpoint_config,
        "metadata": checkpoint.get("metadata", {}),
        "optimizer_loaded": optimizer_loaded,
        "scaler_loaded": scaler_loaded,
        "has_optimizer_state": "optimizer_state_dict" in checkpoint,
        "has_scaler_state": "scaler_state_dict" in checkpoint,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
