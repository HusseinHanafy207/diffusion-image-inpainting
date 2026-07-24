"""Flexible mask generators for inpainting.

Convention (binary mask ``m``):
    1 = known / observed pixel
    0 = missing / to be inpainted

Supported types (Phase 1):
    - center square
    - random rectangle
    - random brush strokes (scratches)
    - random holes (missing pixels)
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

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
        per call (unless ``mask_type`` is passed explicitly).
    """

    def __init__(
        self,
        image_size: int = 28,
        mask_types: Sequence[MaskType | str] | None = None,
    ) -> None:
        self.image_size = image_size
        if mask_types is None:
            mask_types = list(MaskType)
        self.mask_types = [MaskType(t) for t in mask_types]

    def __call__(
        self,
        batch_size: int = 1,
        mask_type: MaskType | str | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return masks of shape ``(B, 1, H, W)`` with values in {0, 1}."""
        raise NotImplementedError("Phase 1: implement MaskGenerator.__call__")

    def center(self, batch_size: int = 1) -> torch.Tensor:
        raise NotImplementedError

    def rectangle(self, batch_size: int = 1) -> torch.Tensor:
        raise NotImplementedError

    def brush(self, batch_size: int = 1) -> torch.Tensor:
        raise NotImplementedError

    def holes(self, batch_size: int = 1) -> torch.Tensor:
        raise NotImplementedError
