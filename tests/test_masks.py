"""Unit tests for MaskGenerator."""

from __future__ import annotations

import torch

from image_inpainting.masks import (
    MaskGenerator,
    MaskType,
    build_mask_generator_from_config,
)


def test_center_mask_shape_and_values() -> None:
    gen = MaskGenerator(image_size=28, center_ratio=0.4)
    masks = gen.center(batch_size=4)
    assert masks.shape == (4, 1, 28, 28)
    assert set(masks.unique().tolist()) <= {0.0, 1.0}
    # Center should contain missing pixels; border should stay known.
    assert masks[:, :, 0, 0].eq(1).all()
    assert masks[:, :, 14, 14].eq(0).all()


def test_center_jitter_moves_hole_but_stays_in_bounds() -> None:
    torch.manual_seed(0)
    gen = MaskGenerator(image_size=64, center_ratio=0.4, center_jitter_ratio=0.15)
    masks = gen.center(batch_size=32)
    assert masks.shape == (32, 1, 64, 64)
    assert set(masks.unique().tolist()) <= {0.0, 1.0}
    # Fixed dead-center hole would be identical across the batch; jitter should vary.
    assert not torch.equal(masks[0], masks[1]) or not torch.equal(masks[2], masks[3])
    side = int(round(0.4 * 64))
    for i in range(masks.shape[0]):
        missing = (masks[i, 0] == 0).nonzero(as_tuple=False)
        assert missing.numel() > 0
        ys, xs = missing[:, 0], missing[:, 1]
        assert int(ys.min()) >= 0 and int(ys.max()) < 64
        assert int(xs.min()) >= 0 and int(xs.max()) < 64
        assert int(ys.max() - ys.min()) + 1 == side
        assert int(xs.max() - xs.min()) + 1 == side


def test_center_zero_jitter_is_identical_across_batch() -> None:
    gen = MaskGenerator(image_size=28, center_ratio=0.4, center_jitter_ratio=0.0)
    masks = gen.center(batch_size=4)
    assert torch.equal(masks[0], masks[1])
    assert torch.equal(masks[0], masks[3])


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


def test_mask_type_weights_force_single_type() -> None:
    torch.manual_seed(3)
    gen = MaskGenerator(
        image_size=28,
        mask_types=[MaskType.CENTER, MaskType.BRUSH],
        mask_type_weights=[1.0, 0.0],
        center_jitter_ratio=0.0,
    )
    ref = gen.center(1)
    for _ in range(20):
        assert torch.equal(gen(1), ref)


def test_mask_type_weights_from_mapping() -> None:
    gen = MaskGenerator(
        image_size=28,
        mask_types=["center", "rectangle", "brush", "holes"],
        mask_type_weights={"center": 0.4, "rectangle": 0.4, "brush": 0.1, "holes": 0.1},
    )
    assert torch.allclose(
        gen.mask_type_probs, torch.tensor([0.4, 0.4, 0.1, 0.1], dtype=torch.float32)
    )


def test_build_mask_generator_phase1_fields() -> None:
    gen = build_mask_generator_from_config(
        {
            "image_size": 64,
            "mask_types": ["center", "rectangle", "brush", "holes"],
            "mask_type_weights": {
                "center": 0.4,
                "rectangle": 0.4,
                "brush": 0.1,
                "holes": 0.1,
            },
            "center_ratio": 0.4,
            "center_jitter_ratio": 0.15,
        }
    )
    assert gen.center_jitter_ratio == 0.15
    assert torch.allclose(
        gen.mask_type_probs, torch.tensor([0.4, 0.4, 0.1, 0.1], dtype=torch.float32)
    )


def test_mask_convention_known_is_one() -> None:
    """1 = known, 0 = missing — majority of a center mask should remain known."""
    gen = MaskGenerator(image_size=28, center_ratio=0.4)
    mask = gen.center(1)[0, 0]
    assert mask.mean().item() > 0.5
