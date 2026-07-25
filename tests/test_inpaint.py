"""Tests for reverse-diffusion inpainting (RePaint resampling)."""

from __future__ import annotations

import torch
from generative_models.ddpm import NoiseScheduler

from image_inpainting.diffusion import inpaint
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
