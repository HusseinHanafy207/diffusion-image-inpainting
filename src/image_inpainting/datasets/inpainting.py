"""Inpainting dataset wrapper.

Wraps a base image dataset (MNIST first, then Fashion-MNIST / CelebA / …)
and applies ``MaskGenerator`` on the fly.

Each item returns:
    original      — clean image ``x``, shape (C, H, W)
    masked_image  — ``x * mask`` (known pixels only; missing set to 0)
    mask          — binary known/missing mask, shape (1, H, W)
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


def apply_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Produce a damaged image: known pixels kept, missing set to 0.

    ``mask`` uses 1 = known, 0 = missing.
    """
    return image * mask


class InpaintingDataset(Dataset):
    """Dataset yielding ``(original, masked_image, mask)``.

    Parameters
    ----------
    base_dataset:
        Underlying dataset whose ``__getitem__`` returns ``(image, label)``
        or ``image`` only.
    mask_generator:
        Instance of :class:`~image_inpainting.masks.MaskGenerator`.
    """

    def __init__(self, base_dataset: Dataset, mask_generator: Any) -> None:
        self.base_dataset = base_dataset
        self.mask_generator = mask_generator

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Phase 2: implement InpaintingDataset.__getitem__")
