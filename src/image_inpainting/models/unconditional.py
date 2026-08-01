"""Unconditional DDPM denoiser for RePaint-style inference.

Training uses a standard ``generative_models.ddpm.DDPM`` (noise ε̂ from ``x_t``
only — no mask channels). At inference I wrap the U-Net so the existing
RePaint ``inpaint()`` API still receives ``(x_t, t, masked_image, mask)`` but
ignores the conditioning tensors.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from generative_models.ddpm import DDPM, NoiseScheduler, UNet


class UncondInpaintAdapter(nn.Module):
    """Adapt an unconditional U-Net / DDPM to the conditioned inpaint call signature.

    ``forward(x_t, t, masked_image, mask)`` → ``ε̂ = unet(x_t, t)``.
    """

    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        del masked_image, mask  # unused — mask constraint is applied in RePaint stitch
        if isinstance(self.denoiser, DDPM):
            return self.denoiser.predict_noise(x_t, t)
        return self.denoiser(x_t, t)


def build_unconditional_ddpm_from_config(config: dict[str, Any]) -> DDPM:
    """Build RGB/grayscale unconditional ``DDPM`` from a YAML config dict."""
    channels = int(config.get("image_channels", 1))
    return DDPM(
        num_timesteps=int(config["num_timesteps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
        in_channels=channels,
        out_channels=channels,
        base_channels=int(config.get("base_channels", 64)),
        channel_mult=tuple(config.get("channel_mult", [1, 2, 4])),
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        attention_resolutions=tuple(config.get("attention_resolutions", [16])),
        dropout=float(config.get("dropout", 0.1)),
        image_size=int(config.get("image_size", 64)),
    )


def build_unconditional_unet_from_config(config: dict[str, Any]) -> UNet:
    """Build bare ``UNet`` (same layout as :func:`build_unconditional_ddpm_from_config`)."""
    channels = int(config.get("image_channels", 1))
    return UNet(
        in_channels=channels,
        out_channels=channels,
        base_channels=int(config.get("base_channels", 64)),
        channel_mult=tuple(config.get("channel_mult", [1, 2, 4])),
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        attention_resolutions=tuple(config.get("attention_resolutions", [16])),
        dropout=float(config.get("dropout", 0.1)),
        image_size=int(config.get("image_size", 64)),
    )


def scheduler_from_config(config: dict[str, Any]) -> NoiseScheduler:
    return NoiseScheduler(
        num_timesteps=int(config["num_timesteps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    )
