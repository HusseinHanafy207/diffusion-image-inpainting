"""Unit tests for MaskGenerator."""

from __future__ import annotations

import torch

from image_inpainting.masks import MaskGenerator, MaskType


def test_center_mask_shape_and_values() -> None:
    gen = MaskGenerator(image_size=28, center_ratio=0.4)
    masks = gen.center(batch_size=4)
    assert masks.shape == (4, 1, 28, 28)
    assert set(masks.unique().tolist()) <= {0.0, 1.0}
    # Center should contain missing pixels; border should stay known.
    assert masks[:, :, 0, 0].eq(1).all()
    assert masks[:, :, 14, 14].eq(0).all()


def test_rectangle_has_missing_region() -> None:
    torch.manual_seed(0)
    gen = MaskGenerator(image_size=28)
    masks = gen.rectangle(batch_size=8)
    assert masks.shape == (8, 1, 28, 28)
    assert (masks == 0).any()
    assert (masks == 1).any()


def test_brush_and_holes_produce_binary_masks() -> None:
    torch.manual_seed(1)
    gen = MaskGenerator(image_size=28)
    brush = gen.brush(batch_size=4)
    holes = gen.holes(batch_size=4)
    for masks in (brush, holes):
        assert masks.shape == (4, 1, 28, 28)
        assert set(masks.unique().tolist()) <= {0.0, 1.0}
        assert (masks == 0).any()


def test_call_fixed_type_and_device() -> None:
    gen = MaskGenerator(image_size=16)
    masks = gen(batch_size=3, mask_type="center", device="cpu")
    assert masks.shape == (3, 1, 16, 16)
    assert masks.device.type == "cpu"


def test_call_random_types_from_subset() -> None:
    torch.manual_seed(2)
    gen = MaskGenerator(image_size=28, mask_types=[MaskType.CENTER, MaskType.HOLES])
    masks = gen(batch_size=16)
    assert masks.shape == (16, 1, 28, 28)
    assert (masks == 0).any()


def test_mask_convention_known_is_one() -> None:
    """1 = known, 0 = missing — majority of a center mask should remain known."""
    gen = MaskGenerator(image_size=28, center_ratio=0.4)
    mask = gen.center(1)[0, 0]
    assert mask.mean().item() > 0.5
