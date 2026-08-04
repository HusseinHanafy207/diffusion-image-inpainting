"""Timestep respacing for shorter reverse schedules (RePaint / guided-diffusion).

Train on a full ``T``-step DDPM, then sample on a subsampled schedule whose
noise levels match selected original timesteps. The U-Net still conditions on
the original ``t`` via ``timestep_map``.

Ported from RePaint ``guided_diffusion/respace.py`` (OpenAI guided-diffusion),
adapted to ``generative_models.ddpm.NoiseScheduler``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch


def space_timesteps(
    num_timesteps: int,
    section_counts: str | int | Sequence[int],
) -> set[int]:
    """Select timesteps from an original process for a shorter reverse chain.

    Parameters
    ----------
    num_timesteps:
        Length of the trained schedule (e.g. 1000).
    section_counts:
        - ``int`` / ``"250"``: one section with that many steps (paper default).
        - ``"10,15,20"``: per-section counts over equal partitions of ``[0, T)``.
        - ``"ddimN"``: fixed integer stride yielding exactly ``N`` steps (DDIM).

    Returns
    -------
    Set of original timestep indices to keep.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {desired_count} steps with an integer "
                f"stride for num_timesteps={num_timesteps}"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    else:
        section_counts = list(section_counts)

    if not section_counts:
        raise ValueError("section_counts must be non-empty")
    if any(c < 1 for c in section_counts):
        raise ValueError(f"section counts must be >= 1, got {section_counts}")

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps: list[int] = []

    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1.0
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps: list[int] = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps.extend(taken_steps)
        start_idx += size

    return set(all_steps)


class SpacedNoiseScheduler:
    """Noise schedule over a subsampled subset of a trained ``T``-step process.

    Local indices ``j ∈ [0, N)`` index the short β chain. ``map_timesteps(j)``
    returns the original trained timestep for the U-Net.
    """

    def __init__(self, base: Any, use_timesteps: set[int] | Sequence[int]) -> None:
        use = set(int(t) for t in use_timesteps)
        if not use:
            raise ValueError("use_timesteps must be non-empty")
        trained_t = int(base.num_timesteps)
        if any(t < 0 or t >= trained_t for t in use):
            raise ValueError(
                f"use_timesteps must be in [0, {trained_t - 1}], got {sorted(use)[:5]}…"
            )

        self.base = base
        self.use_timesteps = use
        self.original_num_steps = trained_t

        # Rebuild betas so each local step matches ᾱ of a kept original step.
        timestep_map: list[int] = []
        new_betas: list[float] = []
        last_alpha_cumprod = 1.0
        alphas_cumprod = base.alphas_cumprod
        for i in range(trained_t):
            if i in use:
                alpha_cumprod = float(alphas_cumprod[i].item())
                new_betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)

        if not new_betas:
            raise ValueError("no timesteps selected for respacing")

        self.timestep_map = timestep_map
        self.num_timesteps = len(new_betas)
        self.beta_start = float(new_betas[0])
        self.beta_end = float(new_betas[-1])
        self.schedule = getattr(base, "schedule", "linear")

        betas = torch.tensor(new_betas, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod_new = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod_new
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod_new)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod_new)

        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64), alphas_cumprod_new[:-1]]
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas

        self._map_tensor = torch.tensor(timestep_map, dtype=torch.long)

    def map_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """Map local schedule indices to original trained timesteps."""
        if t.ndim != 1:
            raise ValueError(f"t must be 1D (batch,), got {tuple(t.shape)}")
        map_tensor = self._map_tensor.to(device=t.device)
        return map_tensor[t.long()]

    def _extract(
        self, coeffs: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]
    ) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must be a 1D tensor of shape (batch,), got {tuple(t.shape)}")
        if t.dtype not in (torch.int32, torch.int64, torch.long):
            raise TypeError(f"t must be an integer tensor, got dtype={t.dtype}")
        if torch.any(t < 0) or torch.any(t >= self.num_timesteps):
            raise ValueError(
                f"t values must be in [0, {self.num_timesteps - 1}], "
                f"got min={int(t.min())}, max={int(t.max())}"
            )
        out = coeffs.to(device=t.device, dtype=torch.float32).gather(0, t.long())
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t.shape[0] != x_0.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x_0 has batch {x_0.shape[0]}, t has {t.shape[0]}"
            )
        if noise is None:
            noise = torch.randn_like(x_0)
        elif noise.shape != x_0.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must match x_0 shape {tuple(x_0.shape)}"
            )
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def p_sample_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise_pred.shape != x_t.shape:
            raise ValueError(
                f"noise_pred shape {tuple(noise_pred.shape)} must match "
                f"x_t shape {tuple(x_t.shape)}"
            )
        if t.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x_t has batch {x_t.shape[0]}, t has {t.shape[0]}"
            )
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        model_mean = sqrt_recip_alpha * (
            x_t - beta_t * noise_pred / sqrt_one_minus_alpha_bar
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        if noise is None:
            noise = torch.randn_like(x_t)
        elif noise.shape != x_t.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must match x_t shape {tuple(x_t.shape)}"
            )
        sigma_t = torch.sqrt(self._extract(self.posterior_variance, t, x_t.shape))
        return model_mean + nonzero_mask * sigma_t * noise


def create_respaced_scheduler(base: Any, num_steps: int) -> SpacedNoiseScheduler:
    """Build a respaced scheduler with ``num_steps`` reverse steps from ``base``."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    trained_t = int(base.num_timesteps)
    if num_steps > trained_t:
        raise ValueError(
            f"num_steps ({num_steps}) cannot exceed trained T={trained_t}"
        )
    if num_steps == trained_t:
        use = set(range(trained_t))
    else:
        use = space_timesteps(trained_t, num_steps)
    return SpacedNoiseScheduler(base, use)
