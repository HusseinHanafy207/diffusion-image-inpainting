"""Dispatch image datasets from a config ``dataset`` name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from image_inpainting.datasets.fashion_mnist import (
    get_fashion_mnist_dataset,
    get_fashion_mnist_inpainting_dataloaders,
)
from image_inpainting.datasets.mnist import (
    get_mnist_dataset,
    get_mnist_inpainting_dataloaders,
)
from image_inpainting.masks.generator import MaskGenerator, MaskType


def normalize_dataset_name(name: str) -> str:
    """``Fashion-MNIST`` / ``fashion_mnist`` → ``fashionmnist``."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def get_base_dataset(
    dataset: str,
    data_dir: str | Path,
    *,
    train: bool = True,
) -> Dataset:
    """Return a torchvision image dataset selected by name."""
    key = normalize_dataset_name(dataset)
    if key == "mnist":
        return get_mnist_dataset(data_dir, train=train)
    if key == "fashionmnist":
        return get_fashion_mnist_dataset(data_dir, train=train)
    raise ValueError(
        f"Unknown dataset {dataset!r}. Supported: MNIST, Fashion-MNIST"
    )


def get_inpainting_dataloaders(
    dataset: str,
    batch_size: int,
    data_dir: str | Path,
    mask_generator: MaskGenerator | None = None,
    *,
    mask_type: MaskType | str | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Train / val loaders for the named dataset."""
    key = normalize_dataset_name(dataset)
    if key == "mnist":
        return get_mnist_inpainting_dataloaders(
            batch_size,
            data_dir,
            mask_generator,
            mask_type=mask_type,
            num_workers=num_workers,
        )
    if key == "fashionmnist":
        return get_fashion_mnist_inpainting_dataloaders(
            batch_size,
            data_dir,
            mask_generator,
            mask_type=mask_type,
            num_workers=num_workers,
        )
    raise ValueError(
        f"Unknown dataset {dataset!r}. Supported: MNIST, Fashion-MNIST"
    )


def get_inpainting_dataloaders_from_config(
    config: dict[str, Any],
    mask_generator: MaskGenerator,
    *,
    mask_type: MaskType | str | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build loaders using ``config['dataset']``, ``batch_size``, ``data_dir``."""
    return get_inpainting_dataloaders(
        str(config.get("dataset", "MNIST")),
        batch_size=int(config["batch_size"]),
        data_dir=config["data_dir"],
        mask_generator=mask_generator,
        mask_type=mask_type,
        num_workers=num_workers,
    )
