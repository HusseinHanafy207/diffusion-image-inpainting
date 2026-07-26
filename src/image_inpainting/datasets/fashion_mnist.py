"""Fashion-MNIST helpers for inpainting experiments."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from image_inpainting.datasets.inpainting import InpaintingDataset
from image_inpainting.datasets.loader_utils import build_dataloader_kwargs
from image_inpainting.masks.generator import MaskGenerator, MaskType


def get_fashion_mnist_dataset(data_dir: str | Path, *, train: bool = True) -> Dataset:
    """Fashion-MNIST with ``ToTensor()`` → float ``(1, 28, 28)`` in ``[0, 1]``."""
    return datasets.FashionMNIST(
        root=str(data_dir),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )


def get_fashion_mnist_inpainting_datasets(
    data_dir: str | Path,
    mask_generator: MaskGenerator | None = None,
    *,
    mask_type: MaskType | str | None = None,
    return_label: bool = False,
) -> tuple[InpaintingDataset, InpaintingDataset]:
    """Train and test :class:`InpaintingDataset` wrappers over Fashion-MNIST."""
    if mask_generator is None:
        mask_generator = MaskGenerator(image_size=28)

    train_ds = InpaintingDataset(
        get_fashion_mnist_dataset(data_dir, train=True),
        mask_generator,
        mask_type=mask_type,
        return_label=return_label,
    )
    test_ds = InpaintingDataset(
        get_fashion_mnist_dataset(data_dir, train=False),
        mask_generator,
        mask_type=mask_type,
        return_label=return_label,
    )
    return train_ds, test_ds


def get_fashion_mnist_inpainting_dataloaders(
    batch_size: int,
    data_dir: str | Path,
    mask_generator: MaskGenerator | None = None,
    *,
    mask_type: MaskType | str | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Train / test loaders yielding ``(original, masked_image, mask)`` batches."""
    train_ds, test_ds = get_fashion_mnist_inpainting_datasets(
        data_dir,
        mask_generator,
        mask_type=mask_type,
    )
    loader_kwargs = build_dataloader_kwargs(
        num_workers=num_workers, pin_memory=pin_memory
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader
