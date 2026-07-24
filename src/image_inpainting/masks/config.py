"""Build MaskGenerator from a YAML config dict."""

from __future__ import annotations

from typing import Any, Sequence

from image_inpainting.masks.generator import MaskGenerator, MaskType


def _as_pair(value: Sequence[int] | tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {value!r}")
    return int(value[0]), int(value[1])


def build_mask_generator_from_config(config: dict[str, Any]) -> MaskGenerator:
    """Create a :class:`MaskGenerator` using inpainting config fields."""
    mask_types = config.get("mask_types")
    return MaskGenerator(
        image_size=int(config.get("image_size", 28)),
        mask_types=mask_types if mask_types is not None else list(MaskType),
        center_ratio=float(config.get("center_ratio", 0.4)),
        rect_min_ratio=float(config.get("rect_min_ratio", 0.2)),
        rect_max_ratio=float(config.get("rect_max_ratio", 0.5)),
        brush_num_strokes=_as_pair(
            config.get("brush_num_strokes", [1, 4]), "brush_num_strokes"
        ),
        brush_thickness=_as_pair(
            config.get("brush_thickness", [1, 3]), "brush_thickness"
        ),
        brush_steps=int(config.get("brush_steps", 12)),
        holes_num=_as_pair(config.get("holes_num", [3, 12]), "holes_num"),
        holes_radius=_as_pair(config.get("holes_radius", [1, 3]), "holes_radius"),
    )
