"""Tests for reverse-diffusion inpainting (RePaint resampling + respacing)."""

from __future__ import annotations

import torch
from generative_models.ddpm import NoiseScheduler

from image_inpainting.diffusion import (
    SpacedNoiseScheduler,
    create_respaced_scheduler,
    inpaint,
    resolve_inpaint_scheduler,
    resolve_inpaint_timesteps,
    space_timesteps,
)
from image_inpainting.diffusion.inpaint_sampler import get_repaint_schedule
from image_inpainting.models import ConditionedUNet


def _tiny_model() -> ConditionedUNet:
    return ConditionedUNet(
        image_channels=1,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(7,),
        dropout=0.0,
        image_size=28,
    )


def test_repaint_schedule_no_resample_is_plain_countdown() -> None:
    times = get_repaint_schedule(num_timesteps=5, jump_length=10, jump_n_sample=1)
    assert times == [4, 3, 2, 1, 0, -1]


def test_repaint_schedule_has_forward_jumps() -> None:
    times = get_repaint_schedule(num_timesteps=20, jump_length=5, jump_n_sample=2)
    assert times[-1] == -1
    # Must contain at least one upward transition.
    assert any(t2 > t1 for t1, t2 in zip(times[:-1], times[1:]) if t2 != -1)
    # Consecutive steps always differ by 1 (except the final -1 sentinel pairing).
    for t1, t2 in zip(times[:-1], times[1:]):
        assert abs(t1 - t2) == 1


def test_space_timesteps_250_of_1000() -> None:
    steps = space_timesteps(1000, 250)
    assert len(steps) == 250
    assert min(steps) == 0
    assert max(steps) == 999


def test_create_respaced_scheduler_maps_to_original_t() -> None:
    base = NoiseScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    spaced = create_respaced_scheduler(base, 250)
    assert isinstance(spaced, SpacedNoiseScheduler)
    assert spaced.num_timesteps == 250
    assert spaced.timestep_map[0] == 0
    assert spaced.timestep_map[-1] == 999
    # ᾱ at local end matches original ᾱ at last mapped step.
    assert torch.isclose(
        spaced.alphas_cumprod[-1],
        base.alphas_cumprod[spaced.timestep_map[-1]],
        rtol=0,
        atol=1e-10,
    )
    t = torch.tensor([0, 1, 249], dtype=torch.long)
    mapped = spaced.map_timesteps(t)
    assert mapped.tolist() == [
        spaced.timestep_map[0],
        spaced.timestep_map[1],
        spaced.timestep_map[249],
    ]


def test_resolve_inpaint_scheduler_respaces_by_default() -> None:
    scheduler = NoiseScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    assert resolve_inpaint_timesteps(scheduler, None) == 1000
    assert resolve_inpaint_timesteps(scheduler, 1000) == 1000
    assert resolve_inpaint_timesteps(scheduler, 250) == 250

    spaced = resolve_inpaint_scheduler(scheduler, 250)
    assert isinstance(spaced, SpacedNoiseScheduler)
    assert spaced.num_timesteps == 250
    assert spaced.timestep_map[-1] == 999

    # Unsafe truncation keeps the original scheduler object.
    truncated = resolve_inpaint_scheduler(
        scheduler, 250, allow_unsafe_timesteps=True
    )
    assert truncated is scheduler
    assert (
        resolve_inpaint_timesteps(scheduler, 250, allow_unsafe_timesteps=True) == 250
    )


def test_inpaint_preserves_known_pixels_with_resampling() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    scheduler = NoiseScheduler(num_timesteps=20, beta_start=1e-4, beta_end=0.02)

    original = torch.rand(2, 1, 28, 28)
    mask = torch.ones(2, 1, 28, 28)
    mask[:, :, 8:20, 8:20] = 0.0
    masked = original * mask

    result = inpaint(
        model,
        scheduler,
        masked,
        mask,
        original=original,
        jump_length=5,
        jump_n_sample=2,
        show_progress=False,
    )

    assert result.shape == original.shape
    known_diff = ((result - original) * mask).abs().max().item()
    assert known_diff < 1e-5


def test_inpaint_jump_n_sample_one_matches_no_jump_path() -> None:
    """With r=1 the schedule is a plain countdown (still noise-matched stitch)."""
    torch.manual_seed(1)
    model = _tiny_model().eval()
    scheduler = NoiseScheduler(num_timesteps=8, beta_start=1e-4, beta_end=0.02)

    original = torch.rand(1, 1, 28, 28)
    mask = torch.ones(1, 1, 28, 28)
    mask[:, :, 10:18, 10:18] = 0.0
    masked = original * mask

    result = inpaint(
        model,
        scheduler,
        masked,
        mask,
        original=original,
        jump_length=10,
        jump_n_sample=1,
    )
    assert ((result - original) * mask).abs().max().item() < 1e-5


def test_inpaint_respaced_timesteps_preserves_known_pixels() -> None:
    """``num_timesteps=N < T`` uses respacing and still preserves known pixels."""
    torch.manual_seed(2)
    model = _tiny_model().eval()
    scheduler = NoiseScheduler(num_timesteps=20, beta_start=1e-4, beta_end=0.02)

    original = torch.rand(1, 1, 28, 28)
    mask = torch.ones(1, 1, 28, 28)
    mask[:, :, 8:16, 8:16] = 0.0
    masked = original * mask

    result = inpaint(
        model,
        scheduler,
        masked,
        mask,
        original=original,
        num_timesteps=10,
        jump_n_sample=1,
    )
    assert result.shape == original.shape
    assert ((result - original) * mask).abs().max().item() < 1e-5


def test_inpaint_unsafe_truncation_still_runs() -> None:
    """Explicit flag keeps the old truncated-schedule ablation path."""
    torch.manual_seed(3)
    model = _tiny_model().eval()
    scheduler = NoiseScheduler(num_timesteps=20, beta_start=1e-4, beta_end=0.02)

    original = torch.rand(1, 1, 28, 28)
    mask = torch.ones(1, 1, 28, 28)
    mask[:, :, 8:16, 8:16] = 0.0
    masked = original * mask

    result = inpaint(
        model,
        scheduler,
        masked,
        mask,
        original=original,
        num_timesteps=10,
        jump_n_sample=1,
        allow_unsafe_timesteps=True,
    )
    assert result.shape == original.shape
    assert ((result - original) * mask).abs().max().item() < 1e-5
