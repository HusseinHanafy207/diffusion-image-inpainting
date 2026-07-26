"""Tests for CelebA wiring (no full dataset download required)."""

from __future__ import annotations

import torch

from image_inpainting.datasets import InpaintingDataset, normalize_dataset_name
from image_inpainting.masks import MaskGenerator, MaskType
from image_inpainting.models import build_conditioned_unet_from_config
from image_inpainting.utils import load_config, tensor_to_display


def test_normalize_celeba() -> None:
    assert normalize_dataset_name("CelebA") == "celeba"
    assert normalize_dataset_name("celeb_a") == "celeba"


def test_build_celeba_conditioned_unet() -> None:
    config = load_config("configs/celeba.yaml")
    model = build_conditioned_unet_from_config(config)
    assert model.image_channels == 3
    assert model.in_channels == 7
    assert model.out_channels == 3

    x_t = torch.randn(2, 3, 64, 64)
    masked = torch.rand(2, 3, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 0.0
    t = torch.tensor([10, 20])
    pred = model(x_t, t, masked, mask)
    assert pred.shape == (2, 3, 64, 64)


def test_rgb_inpainting_dataset_shapes() -> None:
    images = torch.rand(4, 3, 64, 64)
    labels = torch.zeros(4, dtype=torch.long)
    from torch.utils.data import TensorDataset

    ds = InpaintingDataset(
        TensorDataset(images, labels),
        MaskGenerator(image_size=64, mask_types=[MaskType.CENTER]),
        mask_type=MaskType.CENTER,
    )
    original, masked, mask = ds[0]
    assert original.shape == (3, 64, 64)
    assert masked.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert (masked == original * mask).all()


def test_tensor_to_display_rgb_and_gray() -> None:
    gray = torch.rand(1, 28, 28)
    rgb = torch.rand(3, 64, 64)
    assert tensor_to_display(gray).shape == (28, 28)
    assert tensor_to_display(rgb).shape == (64, 64, 3)
