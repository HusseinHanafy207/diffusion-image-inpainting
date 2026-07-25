"""Tests for reverse-diffusion inpainting (Phase 5 / RePaint-style)."""

from __future__ import annotations

import torch
from generative_models.ddpm import NoiseScheduler

from image_inpainting.diffusion import inpaint
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


def test_inpaint_preserves_known_pixels_at_t0() -> None:
    """After the final step, known pixels must equal the clean original."""
    torch.manual_seed(0)
    model = _tiny_model().eval()
    scheduler = NoiseScheduler(num_timesteps=10, beta_start=1e-4, beta_end=0.02)

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
        show_progress=False,
    )

    assert result.shape == original.shape
    known_diff = ((result - original) * mask).abs().max().item()
    assert known_diff < 1e-5
    assert ((result - masked) * (1.0 - mask)).abs().sum().item() > 0.0


def test_inpaint_without_original_uses_masked_known() -> None:
    torch.manual_seed(1)
    model = _tiny_model().eval()
    scheduler = NoiseScheduler(num_timesteps=5, beta_start=1e-4, beta_end=0.02)

    masked = torch.zeros(1, 1, 28, 28)
    masked[:, :, :, :14] = 0.7
    mask = torch.ones(1, 1, 28, 28)
    mask[:, :, :, 14:] = 0.0

    result = inpaint(model, scheduler, masked, mask, original=None)
    left = result[:, :, :, :14]
    assert torch.allclose(left, torch.full_like(left, 0.7), atol=1e-5)


def test_inpaint_starts_from_noise_not_clean_canvas() -> None:
    """Sanity: with T=1 the known region still ends clean after the only step."""
    torch.manual_seed(2)
    model = _tiny_model().eval()
    scheduler = NoiseScheduler(num_timesteps=1, beta_start=1e-4, beta_end=0.02)

    original = torch.ones(1, 1, 28, 28) * 0.4
    mask = torch.ones(1, 1, 28, 28)
    mask[:, :, 10:18, 10:18] = 0.0
    masked = original * mask

    result = inpaint(model, scheduler, masked, mask, original=original)
    assert ((result - original) * mask).abs().max().item() < 1e-5
