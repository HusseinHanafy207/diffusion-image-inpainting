"""Reverse-diffusion inpainting with known-pixel reinsertion.

Key idea (every reverse step):

    x = mask * original + (1 - mask) * generated

Only the missing region is free to change; observed pixels are overwritten
with the ground-truth (or damaged-image) values after each denoising step.

This follows the spirit of RePaint (Lugmayr et al., 2022): reuse a diffusion
model and enforce consistency with known pixels during sampling.

Dependencies reused from ``generative_models.ddpm``:
    - NoiseScheduler / ᾱ_t
    - p_sample (single reverse step)
    - forward_diffuse (optional resampling tricks later)
"""

from __future__ import annotations

from typing import Any

import torch


def inpaint(
    model: Any,
    scheduler: Any,
    masked_image: torch.Tensor,
    mask: torch.Tensor,
    *,
    original: torch.Tensor | None = None,
    num_timesteps: int | None = None,
) -> torch.Tensor:
    """Fill missing regions via reverse diffusion with mask reinsertion.

    Parameters
    ----------
    model:
        Conditioned denoiser ``ε_θ(x_t, t, masked_image, mask)``.
    scheduler:
        ``generative_models.ddpm.NoiseScheduler`` (or compatible).
    masked_image:
        Damaged observation ``x_0 ⊙ m``.
    mask:
        Binary mask (1 = known, 0 = missing).
    original:
        Full clean image if available (for oracle reinsertion during eval).
        If ``None``, known pixels are taken from ``masked_image``.
    num_timesteps:
        Override ``scheduler.num_timesteps`` when set.

    Returns
    -------
    Inpainted image in the same value range as training (e.g. [0, 1]).
    """
    raise NotImplementedError("Phase 5: implement reverse-diffusion inpainting")
