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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    expected_config: dict | None = None,
    map_location: str | torch.device = "cpu",
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

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "config": checkpoint_config,
    }
