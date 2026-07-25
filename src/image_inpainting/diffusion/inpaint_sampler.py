"""Reverse-diffusion inpainting with RePaint-style known-pixel reinsertion.

At every reverse step the known region is replaced by a **noise-matched**
version of the original image, not the clean ``x_0``:

    x_{t-1} = m ⊙ q(x_0, t-1) + (1 - m) ⊙ x_{t-1}^{unknown}

Stitching clean pixels into a noisy canvas creates a distribution the U-Net
never saw during training. Matching noise levels (RePaint, Lugmayr et al.,
2022) keeps the boundary seamless.

Dependencies reused from ``generative_models.ddpm``:
    - NoiseScheduler.p_sample_step
    - NoiseScheduler.q_sample
"""

from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm


@torch.no_grad()
def inpaint(
    model: Any,
    scheduler: Any,
    masked_image: torch.Tensor,
    mask: torch.Tensor,
    *,
    original: torch.Tensor | None = None,
    num_timesteps: int | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Fill missing regions via reverse diffusion with noisy known reinsertion.

    Algorithm (RePaint-style)
    -------------------------
    1. Start from pure Gaussian noise ``x_T ~ N(0, I)``.
    2. For ``t = T-1, …, 0``:
         A. Predict noise and take one reverse step → ``x_{t-1}^{unknown}``
         B. Forward-diffuse the clean known image to level ``t-1``
            → ``x_{t-1}^{known} = q(x_0, t-1)``  (use clean ``x_0`` when ``t = 0``)
         C. Stitch: ``x_{t-1} = m ⊙ x_{t-1}^{known} + (1-m) ⊙ x_{t-1}^{unknown}``
    3. Clamp to ``[0, 1]``.

    Parameters
    ----------
    model:
        Conditioned denoiser ``ε_θ(x_t, t, masked_image, mask)``.
    scheduler:
        ``generative_models.ddpm.NoiseScheduler`` (or compatible).
    masked_image:
        Damaged observation ``x_0 ⊙ m``, shape ``(B, C, H, W)``.
    mask:
        Binary mask (1 = known, 0 = missing), shape ``(B, 1, H, W)``.
    original:
        Full clean image. Required for proper known-region resampling.
        If ``None``, falls back to ``masked_image`` (known pixels only).
    num_timesteps:
        Override ``scheduler.num_timesteps`` when set.
    show_progress:
        Show a tqdm bar over reverse steps.

    Returns
    -------
    Inpainted image in ``[0, 1]``, shape ``(B, C, H, W)``.
    """
    if masked_image.ndim != 4:
        raise ValueError(f"masked_image must be (B, C, H, W), got {tuple(masked_image.shape)}")
    if mask.shape[0] != masked_image.shape[0] or mask.shape[-2:] != masked_image.shape[-2:]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} incompatible with "
            f"masked_image {tuple(masked_image.shape)}"
        )
    if mask.shape[1] != 1:
        raise ValueError(f"mask must have 1 channel, got shape {tuple(mask.shape)}")

    model.eval()
    device = masked_image.device
    known_clean = original if original is not None else masked_image
    if known_clean.shape != masked_image.shape:
        raise ValueError(
            f"original shape {tuple(known_clean.shape)} must match "
            f"masked_image {tuple(masked_image.shape)}"
        )

    known_clean = known_clean.to(device)
    mask = mask.to(device)
    masked_image = masked_image.to(device)

    # Start from pure noise (standard DDPM / RePaint initialization).
    x = torch.randn_like(known_clean)

    steps = int(num_timesteps if num_timesteps is not None else scheduler.num_timesteps)
    timesteps = range(steps - 1, -1, -1)
    iterator = tqdm(timesteps, desc="inpainting", leave=False) if show_progress else timesteps

    for t in iterator:
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

        # A — denoise the full canvas (unknown region will be kept).
        noise_pred = model(x, t_batch, masked_image, mask)
        x_unknown = scheduler.p_sample_step(x, t_batch, noise_pred)

        # B — known region at the same noise level as x_unknown (timestep t-1).
        if t > 0:
            t_prev = torch.full((x.shape[0],), t - 1, device=device, dtype=torch.long)
            x_known = scheduler.q_sample(known_clean, t_prev)
        else:
            x_known = known_clean

        # C — stitch noise-matched known pixels with generated unknown pixels.
        x = mask * x_known + (1.0 - mask) * x_unknown

    return x.clamp(0.0, 1.0)
