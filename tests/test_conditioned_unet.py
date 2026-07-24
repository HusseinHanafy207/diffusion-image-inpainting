"""Tests for mask-conditioned U-Net (Phase 3)."""

from __future__ import annotations

import torch
import pytest

from image_inpainting.models import ConditionedUNet, build_conditioned_unet_from_config
from image_inpainting.utils import load_config


def _tiny_model() -> ConditionedUNet:
    """Small U-Net so tests stay fast."""
    return ConditionedUNet(
        image_channels=1,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(7,),
        dropout=0.0,
        image_size=28,
    )


def test_conditioned_unet_input_channels() -> None:
    model = _tiny_model()
    assert model.in_channels == 3  # x_t + masked + mask
    assert model.out_channels == 1
    assert model.unet.in_channels == 3


def test_forward_shape_and_grad() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    b = 4
    x_t = torch.randn(b, 1, 28, 28)
    masked = torch.randn(b, 1, 28, 28)
    mask = torch.ones(b, 1, 28, 28)
    mask[:, :, 10:18, 10:18] = 0.0
    t = torch.randint(0, 1000, (b,))

    pred = model(x_t, t, masked, mask)
    assert pred.shape == (b, 1, 28, 28)

    loss = pred.pow(2).mean()
    loss.backward()
    assert model.unet.conv_in.weight.grad is not None
    # First conv must see 3 input channels.
    assert model.unet.conv_in.weight.shape[1] == 3


def test_rejects_wrong_mask_shape() -> None:
    model = _tiny_model()
    x_t = torch.randn(2, 1, 28, 28)
    masked = torch.randn(2, 1, 28, 28)
    bad_mask = torch.ones(2, 1, 14, 14)
    t = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="mask shape"):
        model(x_t, t, masked, bad_mask)


def test_rejects_manual_in_channels() -> None:
    with pytest.raises(TypeError, match="image_channels"):
        ConditionedUNet(image_channels=1, in_channels=3)


def test_build_from_mnist_config() -> None:
    config = load_config("configs/mnist.yaml")
    model = build_conditioned_unet_from_config(config)
    assert model.in_channels == 3
    assert model.unet.base_channels == 64

    x_t = torch.randn(2, 1, 28, 28)
    masked = torch.randn(2, 1, 28, 28)
    mask = torch.ones(2, 1, 28, 28)
    t = torch.tensor([0, 999])
    with torch.no_grad():
        pred = model(x_t, t, masked, mask)
    assert pred.shape == (2, 1, 28, 28)


def test_timestep_conditioning_changes_output() -> None:
    torch.manual_seed(1)
    model = _tiny_model().eval()
    x_t = torch.randn(1, 1, 28, 28)
    masked = torch.zeros(1, 1, 28, 28)
    mask = torch.ones(1, 1, 28, 28)
    with torch.no_grad():
        y0 = model(x_t, torch.tensor([0]), masked, mask)
        y1 = model(x_t, torch.tensor([999]), masked, mask)
    assert (y0 - y1).abs().mean().item() > 0.0
