"""Load trained inpainting checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from generative_models.ddpm import NoiseScheduler
from generative_models.utils.device import get_device

from image_inpainting.models import ConditionedUNet, build_conditioned_unet_from_config
from image_inpainting.utils import load_config


def build_scheduler_from_config(config: dict[str, Any]) -> NoiseScheduler:
    return NoiseScheduler(
        num_timesteps=int(config["num_timesteps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    )


def load_inpainting_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    config_path: str | Path | None = None,
) -> tuple[ConditionedUNet, NoiseScheduler, dict[str, Any], dict[str, Any]]:
    """Load ``ConditionedUNet`` + scheduler from a training checkpoint.

    Returns
    -------
    model, scheduler, config, checkpoint_dict
    """
    device = device or get_device()
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config")
    if config is None and config_path is not None:
        config = load_config(config_path)
    if config is None:
        raise ValueError(
            "Config not found in checkpoint and no config_path provided."
        )

    model = build_conditioned_unet_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scheduler = build_scheduler_from_config(config)
    return model, scheduler, config, checkpoint
