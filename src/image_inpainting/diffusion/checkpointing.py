"""Load trained inpainting / unconditional DDPM checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from generative_models.ddpm import NoiseScheduler
from generative_models.utils.device import get_device

from image_inpainting.models import ConditionedUNet, build_conditioned_unet_from_config
from image_inpainting.models.unconditional import (
    UncondInpaintAdapter,
    build_unconditional_ddpm_from_config,
)
from image_inpainting.utils import load_config


def build_scheduler_from_config(config: dict[str, Any]) -> NoiseScheduler:
    return NoiseScheduler(
        num_timesteps=int(config["num_timesteps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    )


def _resolve_config(
    checkpoint: dict[str, Any],
    config_path: str | Path | None,
) -> dict[str, Any]:
    config = checkpoint.get("config")
    if config is None and config_path is not None:
        config = load_config(config_path)
    if config is None:
        raise ValueError(
            "Config not found in checkpoint and no config_path provided."
        )
    return config


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
    config = _resolve_config(checkpoint, config_path)

    model = build_conditioned_unet_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scheduler = build_scheduler_from_config(config)
    return model, scheduler, config, checkpoint


def load_unconditional_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    config_path: str | Path | None = None,
) -> tuple[UncondInpaintAdapter, NoiseScheduler, dict[str, Any], dict[str, Any]]:
    """Load unconditional ``DDPM`` wrapped for RePaint ``inpaint()``.

    Returns an :class:`UncondInpaintAdapter` so ``inpaint`` can keep calling
    ``model(x_t, t, masked_image, mask)`` while the U-Net ignores conditioning.
    """
    device = device or get_device()
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = _resolve_config(checkpoint, config_path)
    config.setdefault("training_mode", "unconditional")

    ddpm = build_unconditional_ddpm_from_config(config)
    ddpm.load_state_dict(checkpoint["model_state_dict"])
    ddpm.to(device)
    ddpm.eval()

    # Prefer the DDPM's own scheduler (buffers match training).
    scheduler = ddpm.scheduler
    adapter = UncondInpaintAdapter(ddpm).to(device).eval()
    return adapter, scheduler, config, checkpoint


def load_checkpoint_for_inpaint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    config_path: str | Path | None = None,
    *,
    unconditional: bool | None = None,
) -> tuple[torch.nn.Module, NoiseScheduler, dict[str, Any], dict[str, Any]]:
    """Load conditioned or unconditional weights for RePaint.

    If ``unconditional`` is ``None``, uses ``config['training_mode']`` when set;
    otherwise defaults to conditioned (legacy checkpoints).
    """
    device = device or get_device()
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = _resolve_config(checkpoint, config_path)

    if unconditional is None:
        unconditional = str(config.get("training_mode", "")).lower() == "unconditional"

    if unconditional:
        return load_unconditional_checkpoint(
            checkpoint_path, device=device, config_path=config_path
        )
    return load_inpainting_checkpoint(
        checkpoint_path, device=device, config_path=config_path
    )
