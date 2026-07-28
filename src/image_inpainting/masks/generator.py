"""Flexible mask generators for inpainting.

Convention (binary mask ``m``):
    1 = known / observed pixel
    0 = missing / to be inpainted

Supported types:
    - center square (optional spatial jitter)
    - random rectangle
    - random brush strokes (scratches)
    - random holes (missing pixels)
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Sequence

import torch


class MaskType(str, Enum):
    CENTER = "center"
    RECTANGLE = "rectangle"
    BRUSH = "brush"
    HOLES = "holes"


class MaskGenerator:
    """Create binary known/missing masks for a batch of images.

    Parameters
    ----------
    image_size:
        Height and width of square images (e.g. 28 for MNIST).
    mask_types:
        Types to sample from. If several are given, one is chosen at random
        per sample (unless ``mask_type`` is passed explicitly to ``__call__``).
    mask_type_weights:
        Optional relative sampling weights aligned with ``mask_types``, or a
        mapping ``{type_name: weight}``. ``None`` → uniform over ``mask_types``.
    center_ratio:
        Side length of the center hole as a fraction of ``image_size``.
    center_jitter_ratio:
        Max per-axis offset of the center hole as a fraction of ``image_size``.
        ``0`` keeps a fixed dead-center square; ``>0`` jitters independently
        per sample (clamped so the hole stays inside the image).
    rect_min_ratio, rect_max_ratio:
        Random rectangle hole side length as a fraction of ``image_size``.
    brush_num_strokes:
        Inclusive ``(min, max)`` number of strokes per mask.
    brush_thickness:
        Inclusive ``(min, max)`` stroke radius in pixels.
    brush_steps:
        Number of random-walk steps per stroke.
    holes_num:
        Inclusive ``(min, max)`` number of circular holes.
    holes_radius:
        Inclusive ``(min, max)`` hole radius in pixels.
    """

    def __init__(
        self,
        image_size: int = 28,
        mask_types: Sequence[MaskType | str] | None = None,
        *,
        mask_type_weights: Sequence[float] | Mapping[str, float] | None = None,
        center_ratio: float = 0.4,
        center_jitter_ratio: float = 0.0,
        rect_min_ratio: float = 0.2,
        rect_max_ratio: float = 0.5,
        brush_num_strokes: tuple[int, int] = (1, 4),
        brush_thickness: tuple[int, int] = (1, 3),
        brush_steps: int = 12,
        holes_num: tuple[int, int] = (3, 12),
        holes_radius: tuple[int, int] = (1, 3),
    ) -> None:
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        if not 0.0 < center_ratio < 1.0:
            raise ValueError(f"center_ratio must be in (0, 1), got {center_ratio}")
        if center_jitter_ratio < 0.0:
            raise ValueError(
                f"center_jitter_ratio must be >= 0, got {center_jitter_ratio}"
            )
        if not 0.0 < rect_min_ratio <= rect_max_ratio < 1.0:
            raise ValueError(
                f"Need 0 < rect_min_ratio <= rect_max_ratio < 1, "
                f"got {rect_min_ratio}, {rect_max_ratio}"
            )

        self.image_size = image_size
        if mask_types is None:
            mask_types = list(MaskType)
        self.mask_types = [MaskType(t) for t in mask_types]
        if not self.mask_types:
            raise ValueError("mask_types must be non-empty")

        self.mask_type_probs = self._normalize_type_weights(mask_type_weights)
        self.center_ratio = center_ratio
        self.center_jitter_ratio = center_jitter_ratio
        self.rect_min_ratio = rect_min_ratio
        self.rect_max_ratio = rect_max_ratio
        self.brush_num_strokes = brush_num_strokes
        self.brush_thickness = brush_thickness
        self.brush_steps = brush_steps
        self.holes_num = holes_num
        self.holes_radius = holes_radius

    def __call__(
        self,
        batch_size: int = 1,
        mask_type: MaskType | str | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return masks of shape ``(B, 1, H, W)`` with values in {0, 1}."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if mask_type is not None:
            chosen = MaskType(mask_type)
            masks = self._generate_type(chosen, batch_size)
        else:
            chunks = [
                self._generate_type(self._sample_mask_type(), 1)
                for _ in range(batch_size)
            ]
            masks = torch.cat(chunks, dim=0)

        if device is not None:
            masks = masks.to(device)
        return masks

    def center(self, batch_size: int = 1) -> torch.Tensor:
        """Near-center square hole; optional per-sample spatial jitter."""
        h = w = self.image_size
        side = max(1, int(round(self.center_ratio * h)))
        base_top = (h - side) // 2
        base_left = (w - side) // 2
        max_shift = int(round(self.center_jitter_ratio * h))

        masks = self._ones(batch_size)
        for i in range(batch_size):
            if max_shift > 0:
                dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            else:
                dy = dx = 0
            top = min(max(base_top + dy, 0), h - side)
            left = min(max(base_left + dx, 0), w - side)
            masks[i, :, top : top + side, left : left + side] = 0.0
        return masks

    def rectangle(self, batch_size: int = 1) -> torch.Tensor:
        """Axis-aligned rectangle hole with random size and position per sample."""
        h = w = self.image_size
        masks = self._ones(batch_size)
        for i in range(batch_size):
            side_h = self._rand_int_ratio(self.rect_min_ratio, self.rect_max_ratio, h)
            side_w = self._rand_int_ratio(self.rect_min_ratio, self.rect_max_ratio, w)
            top = int(torch.randint(0, h - side_h + 1, (1,)).item())
            left = int(torch.randint(0, w - side_w + 1, (1,)).item())
            masks[i, :, top : top + side_h, left : left + side_w] = 0.0
        return masks

    def brush(self, batch_size: int = 1) -> torch.Tensor:
        """Random brush strokes (random walks) to simulate scratches."""
        h = w = self.image_size
        masks = self._ones(batch_size)
        min_strokes, max_strokes = self.brush_num_strokes
        min_thick, max_thick = self.brush_thickness

        for i in range(batch_size):
            n_strokes = int(torch.randint(min_strokes, max_strokes + 1, (1,)).item())
            for _ in range(n_strokes):
                thickness = int(torch.randint(min_thick, max_thick + 1, (1,)).item())
                y = int(torch.randint(0, h, (1,)).item())
                x = int(torch.randint(0, w, (1,)).item())
                for _ in range(self.brush_steps):
                    self._punch_disk(masks[i, 0], y, x, thickness)
                    dy = int(torch.randint(-2, 3, (1,)).item())
                    dx = int(torch.randint(-2, 3, (1,)).item())
                    y = min(max(y + dy, 0), h - 1)
                    x = min(max(x + dx, 0), w - 1)
        return masks

    def holes(self, batch_size: int = 1) -> torch.Tensor:
        """Scattered circular holes to simulate missing pixels."""
        h = w = self.image_size
        masks = self._ones(batch_size)
        min_n, max_n = self.holes_num
        min_r, max_r = self.holes_radius

        for i in range(batch_size):
            n_holes = int(torch.randint(min_n, max_n + 1, (1,)).item())
            for _ in range(n_holes):
                radius = int(torch.randint(min_r, max_r + 1, (1,)).item())
                y = int(torch.randint(0, h, (1,)).item())
                x = int(torch.randint(0, w, (1,)).item())
                self._punch_disk(masks[i, 0], y, x, radius)
        return masks

    def _generate_type(self, mask_type: MaskType, batch_size: int) -> torch.Tensor:
        if mask_type is MaskType.CENTER:
            return self.center(batch_size)
        if mask_type is MaskType.RECTANGLE:
            return self.rectangle(batch_size)
        if mask_type is MaskType.BRUSH:
            return self.brush(batch_size)
        if mask_type is MaskType.HOLES:
            return self.holes(batch_size)
        raise ValueError(f"Unknown mask type: {mask_type}")

    def _sample_mask_type(self) -> MaskType:
        idx = int(torch.multinomial(self.mask_type_probs, num_samples=1).item())
        return self.mask_types[idx]

    def _normalize_type_weights(
        self,
        weights: Sequence[float] | Mapping[str, float] | None,
    ) -> torch.Tensor:
        n = len(self.mask_types)
        if weights is None:
            probs = torch.ones(n, dtype=torch.float64)
        elif isinstance(weights, Mapping):
            probs = torch.tensor(
                [float(weights.get(t.value, 0.0)) for t in self.mask_types],
                dtype=torch.float64,
            )
        else:
            if len(weights) != n:
                raise ValueError(
                    f"mask_type_weights length {len(weights)} must match "
                    f"mask_types length {n}"
                )
            probs = torch.tensor([float(w) for w in weights], dtype=torch.float64)

        if (probs < 0).any():
            raise ValueError(f"mask_type_weights must be non-negative, got {weights!r}")
        total = float(probs.sum().item())
        if total <= 0.0:
            raise ValueError(
                f"mask_type_weights must sum to a positive value, got {weights!r}"
            )
        return (probs / total).to(dtype=torch.float32)

    def _ones(self, batch_size: int) -> torch.Tensor:
        h = w = self.image_size
        return torch.ones(batch_size, 1, h, w, dtype=torch.float32)

    def _rand_int_ratio(self, low: float, high: float, size: int) -> int:
        ratio = float(torch.empty(1).uniform_(low, high).item())
        return max(1, min(size, int(round(ratio * size))))

    @staticmethod
    def _punch_disk(mask_hw: torch.Tensor, cy: int, cx: int, radius: int) -> None:
        """Set a circular region to 0 (missing) in-place on a single ``(H, W)`` mask."""
        h, w = mask_hw.shape
        y0 = max(cy - radius, 0)
        y1 = min(cy + radius + 1, h)
        x0 = max(cx - radius, 0)
        x1 = min(cx + radius + 1, w)
        if y0 >= y1 or x0 >= x1:
            return

        yy = torch.arange(y0, y1, device=mask_hw.device).view(-1, 1)
        xx = torch.arange(x0, x1, device=mask_hw.device).view(1, -1)
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        region = mask_hw[y0:y1, x0:x1]
        region[disk] = 0.0
