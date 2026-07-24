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

from image_inpainting.masks.generator import MaskGenerator, MaskType


def apply_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Produce a damaged image: known pixels kept, missing set to 0.

    ``mask`` uses 1 = known, 0 = missing.

    Supports:
        image (C, H, W)  × mask (1, H, W)
        image (B, C, H, W) × mask (B, 1, H, W)  (broadcast on channel)
    """
    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError(
            f"Spatial size mismatch: image {tuple(image.shape)} vs mask {tuple(mask.shape)}"
        )
    return image * mask


def _extract_image(item: Any) -> tuple[torch.Tensor, Any | None]:
    """Pull ``(image, optional_label)`` out of a dataset sample."""
    label: Any | None
    if isinstance(item, (tuple, list)):
        image = item[0]
        label = item[1] if len(item) > 1 else None
    else:
        image = item
        label = None

    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"Expected image tensor from base dataset, got {type(image).__name__}. "
            "Apply torchvision transforms.ToTensor() in the base dataset."
        )
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (C, H, W), got {tuple(image.shape)}")
    return image, label


class InpaintingDataset(Dataset):
    """Dataset yielding ``(original, masked_image, mask)``.

    On every access a fresh mask is drawn (unless ``mask_type`` is fixed),
    then the damaged image is formed with :func:`apply_mask`.

    Parameters
    ----------
    base_dataset:
        Underlying dataset whose ``__getitem__`` returns ``(image, label)``
        or ``image`` only. ``image`` must be a float tensor ``(C, H, W)``.
    mask_generator:
        Instance of :class:`~image_inpainting.masks.MaskGenerator`.
    mask_type:
        If set, always use this mask family; otherwise sample from the
        generator's configured ``mask_types``.
    return_label:
        If ``True``, also return the base dataset label as a 4th element.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        mask_generator: MaskGenerator,
        *,
        mask_type: MaskType | str | None = None,
        return_label: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.mask_generator = mask_generator
        self.mask_type = MaskType(mask_type) if mask_type is not None else None
        self.return_label = return_label

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(
        self, index: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ):
        image, label = _extract_image(self.base_dataset[index])
        mask = self.mask_generator(batch_size=1, mask_type=self.mask_type)[0]
        # mask: (1, H, W); image: (C, H, W)
        if mask.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                f"Mask size {tuple(mask.shape[-2:])} does not match "
                f"image size {tuple(image.shape[-2:])}. "
                "Set MaskGenerator.image_size to the dataset resolution."
            )
        masked_image = apply_mask(image, mask)

        if self.return_label:
            return image, masked_image, mask, label
        return image, masked_image, mask
