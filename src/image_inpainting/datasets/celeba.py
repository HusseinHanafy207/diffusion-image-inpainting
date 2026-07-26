"""CelebA helpers for inpainting experiments (RGB faces)."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from image_inpainting.datasets.inpainting import InpaintingDataset
from image_inpainting.masks.generator import MaskGenerator, MaskType


def _celeba_transform(image_size: int) -> transforms.Compose:
    # Standard CelebA face crop then resize (aligned roughly on the face).
    return transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )


def get_celeba_dataset(
    data_dir: str | Path,
    *,
    train: bool = True,
    image_size: int = 64,
    download: bool = True,
) -> Dataset:
    """CelebA RGB images as float ``(3, H, W)`` in ``[0, 1]``.

    Uses torchvision ``split='train'`` / ``'valid'``. CelebA is large (~1.4GB);
    the first download may require a working Google Drive fetch from torchvision.
    """
    split = "train" if train else "valid"
    return datasets.CelebA(
        root=str(data_dir),
        split=split,
        target_type="attr",
        transform=_celeba_transform(image_size),
        download=download,
    )


def get_celeba_inpainting_datasets(
    data_dir: str | Path,
    mask_generator: MaskGenerator | None = None,
    *,
    image_size: int = 64,
    mask_type: MaskType | str | None = None,
    return_label: bool = False,
    download: bool = True,
) -> tuple[InpaintingDataset, InpaintingDataset]:
    """Train and validation :class:`InpaintingDataset` wrappers over CelebA."""
    if mask_generator is None:
        mask_generator = MaskGenerator(image_size=image_size)

    train_ds = InpaintingDataset(
        get_celeba_dataset(
            data_dir, train=True, image_size=image_size, download=download
        ),
        mask_generator,
        mask_type=mask_type,
        return_label=return_label,
    )
    val_ds = InpaintingDataset(
        get_celeba_dataset(
            data_dir, train=False, image_size=image_size, download=download
        ),
        mask_generator,
        mask_type=mask_type,
        return_label=return_label,
    )
    return train_ds, val_ds


def get_celeba_inpainting_dataloaders(
    batch_size: int,
    data_dir: str | Path,
    mask_generator: MaskGenerator | None = None,
    *,
    image_size: int = 64,
    mask_type: MaskType | str | None = None,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Train / val loaders yielding ``(original, masked_image, mask)`` batches."""
    train_ds, val_ds = get_celeba_inpainting_datasets(
        data_dir,
        mask_generator,
        image_size=image_size,
        mask_type=mask_type,
        download=download,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
