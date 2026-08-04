"""Reverse-diffusion inpainting with RePaint resampling.

Core stitch (noise-matched known pixels):

    x_{t-1} = m ⊙ q(x_0, t-1) + (1 - m) ⊙ x_{t-1}^{unknown}

Resampling (Lugmayr et al., 2022): after denoising, jump **forward** in
diffusion time by ``jump_length`` steps (add noise), then denoise again.
Repeating this ``jump_n_sample`` times lets the U-Net re-harmonize the hole
with the known region.

Timestep respacing: ``num_timesteps=N < T`` (without ``allow_unsafe_timesteps``)
builds a short schedule whose noise levels match selected steps of the trained
``T``-step process (RePaint / guided-diffusion). The U-Net still sees original
timesteps via ``SpacedNoiseScheduler.map_timesteps``.

Dependencies reused from ``generative_models.ddpm``:
    - NoiseScheduler.p_sample_step
    - NoiseScheduler.q_sample
"""

from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from image_inpainting.diffusion.respace import create_respaced_scheduler


def get_repaint_schedule(
    num_timesteps: int,
    jump_length: int = 10,
    jump_n_sample: int = 10,
) -> list[int]:
    """Build the RePaint time schedule (paper Appendix / official code).

    Returns a list of diffusion times ending with ``-1``. Consecutive values
    differ by ±1: decreasing = reverse step, increasing = forward (undo).
    """
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
    if jump_length < 1:
        raise ValueError(f"jump_length must be >= 1, got {jump_length}")
    if jump_n_sample < 1:
        raise ValueError(f"jump_n_sample must be >= 1, got {jump_n_sample}")

    jumps: dict[int, int] = {}
    for j in range(0, num_timesteps - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    t = num_timesteps
    times: list[int] = []
    while t >= 1:
        t = t - 1
        times.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t = t + 1
                times.append(t)
    times.append(-1)
    return times


def _undo_step(scheduler: Any, x: torch.Tensor, t_next: int) -> torch.Tensor:
    """One forward DDPM transition: ``x_t → x_{t+1}`` using ``β_{t+1}``.

    ``t_next`` is the destination timestep index (0-based).
    """
    if t_next < 1 or t_next >= scheduler.num_timesteps:
        raise ValueError(
            f"t_next must be in [1, {scheduler.num_timesteps - 1}], got {t_next}"
        )
    beta = scheduler.betas[t_next].to(device=x.device, dtype=torch.float32)
    noise = torch.randn_like(x)
    return torch.sqrt(1.0 - beta) * x + torch.sqrt(beta) * noise


def resolve_inpaint_scheduler(
    scheduler: Any,
    num_timesteps: int | None,
    *,
    allow_unsafe_timesteps: bool = False,
) -> Any:
    """Resolve the schedule used for reverse sampling.

    - ``None`` or ``== T``: use the trained scheduler as-is.
    - ``N < T`` without ``allow_unsafe_timesteps``: RePaint-style **respacing**
      (short β chain + ``timestep_map`` for the U-Net).
    - ``N < T`` with ``allow_unsafe_timesteps``: truncated first-``N`` indices
      of the trained schedule (ablation only; causes seam artifacts).
    """
    trained_t = int(scheduler.num_timesteps)
    if num_timesteps is None:
        return scheduler
    steps = int(num_timesteps)
    if steps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {steps}")
    if steps > trained_t:
        raise ValueError(
            f"num_timesteps ({steps}) cannot exceed scheduler.num_timesteps ({trained_t})"
        )
    if steps == trained_t:
        return scheduler
    if allow_unsafe_timesteps:
        return scheduler
    return create_respaced_scheduler(scheduler, steps)


def resolve_inpaint_timesteps(
    scheduler: Any,
    num_timesteps: int | None,
    *,
    allow_unsafe_timesteps: bool = False,
) -> int:
    """Resolve how many reverse steps to run (local schedule length).

    Prefer :func:`resolve_inpaint_scheduler` when you also need the schedule
    object. This helper remains for callers that only need the step count.

    With respacing (default for ``N < T``), returns ``N``.
    With ``allow_unsafe_timesteps``, returns ``N`` for truncated ablation.
    """
    trained_t = int(scheduler.num_timesteps)
    if num_timesteps is None:
        return trained_t
    steps = int(num_timesteps)
    if steps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {steps}")
    if steps > trained_t:
        raise ValueError(
            f"num_timesteps ({steps}) cannot exceed scheduler.num_timesteps ({trained_t})"
        )
    if steps < trained_t and allow_unsafe_timesteps:
        return steps
    if steps < trained_t:
        # Respacing path: local length is N.
        return steps
    return trained_t


def _model_timesteps(scheduler: Any, t_batch: torch.Tensor) -> torch.Tensor:
    """Local schedule indices → U-Net timesteps (identity unless respaced)."""
    map_fn = getattr(scheduler, "map_timesteps", None)
    if map_fn is None:
        return t_batch
    return map_fn(t_batch)


def _reverse_and_stitch(
    model: Any,
    scheduler: Any,
    x: torch.Tensor,
    t: int,
    masked_image: torch.Tensor,
    mask: torch.Tensor,
    known_clean: torch.Tensor,
) -> torch.Tensor:
    """Denoise at local timestep ``t`` and stitch noise-matched known pixels."""
    t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    model_t = _model_timesteps(scheduler, t_batch)
    noise_pred = model(x, model_t, masked_image, mask)
    x_unknown = scheduler.p_sample_step(x, t_batch, noise_pred)

    if t > 0:
        t_prev = torch.full((x.shape[0],), t - 1, device=x.device, dtype=torch.long)
        x_known = scheduler.q_sample(known_clean, t_prev)
    else:
        x_known = known_clean

    return mask * x_known + (1.0 - mask) * x_unknown


@torch.no_grad()
def inpaint(
    model: Any,
    scheduler: Any,
    masked_image: torch.Tensor,
    mask: torch.Tensor,
    *,
    original: torch.Tensor | None = None,
    num_timesteps: int | None = None,
    jump_length: int = 10,
    jump_n_sample: int = 10,
    show_progress: bool = False,
    allow_unsafe_timesteps: bool = False,
) -> torch.Tensor:
    """Fill missing regions via RePaint reverse diffusion + resampling.

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
        Full clean image for known-region resampling.
        If ``None``, falls back to ``masked_image``.
    num_timesteps:
        ``None`` (full trained ``T``), or ``N <= T``. When ``N < T`` without
        ``allow_unsafe_timesteps``, uses RePaint-style respacing. With the
        flag, truncates to the first ``N`` trained indices (ablation only).
    jump_length:
        Forward jump size ``j`` (paper default 10). Set ``jump_n_sample=1``
        to disable resampling.
    jump_n_sample:
        Resample count ``r`` (paper default 10). ``1`` = no extra jumps.
    show_progress:
        Show a tqdm bar over schedule transitions.
    allow_unsafe_timesteps:
        Use truncated (non-respaced) schedule for ablation only.

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

    x = torch.randn_like(known_clean)

    trained_t = int(scheduler.num_timesteps)
    use_scheduler = resolve_inpaint_scheduler(
        scheduler,
        num_timesteps,
        allow_unsafe_timesteps=allow_unsafe_timesteps,
    )
    if allow_unsafe_timesteps and num_timesteps is not None and int(num_timesteps) < trained_t:
        steps = int(num_timesteps)
    else:
        steps = int(use_scheduler.num_timesteps)

    times = get_repaint_schedule(steps, jump_length=jump_length, jump_n_sample=jump_n_sample)
    pairs = list(zip(times[:-1], times[1:]))
    iterator = tqdm(pairs, desc="inpainting", leave=False) if show_progress else pairs

    for t_last, t_cur in iterator:
        if t_cur < t_last:
            # Reverse: denoise at t_last, stitch known at t_last-1.
            x = _reverse_and_stitch(
                model, use_scheduler, x, t_last, masked_image, mask, known_clean
            )
        else:
            # Forward jump: add noise t_last → t_cur (always +1 in the schedule).
            x = _undo_step(use_scheduler, x, t_next=t_cur)

    return x.clamp(0.0, 1.0)
