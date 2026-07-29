"""Dispatch image datasets from a config ``dataset`` name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from image_inpainting.datasets.celeba import (
    get_celeba_dataset,
    get_celeba_inpainting_dataloaders,
)
from image_inpainting.datasets.fashion_mnist import (
    get_fashion_mnist_dataset,
    get_fashion_mnist_inpainting_dataloaders,
)
from image_inpainting.datasets.mnist import (
    get_mnist_dataset,
    get_mnist_inpainting_dataloaders,
)
from image_inpainting.masks.generator import MaskGenerator, MaskType

_SUPPORTED = "MNIST, Fashion-MNIST, CelebA"


def normalize_dataset_name(name: str) -> str:
    """``Fashion-MNIST`` / ``fashion_mnist`` → ``fashionmnist``."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def get_base_dataset(
    dataset: str,
    data_dir: str | Path,
    *,
    train: bool = True,
    image_size: int | None = None,
    download: bool = True,
) -> Dataset:
    """Return a torchvision image dataset selected by name."""
    key = normalize_dataset_name(dataset)
    if key == "mnist":
        return get_mnist_dataset(data_dir, train=train)
    if key == "fashionmnist":
        return get_fashion_mnist_dataset(data_dir, train=train)
    if key == "celeba":
        size = 64 if image_size is None else image_size
        return get_celeba_dataset(
            data_dir, train=train, image_size=size, download=download
        )
    raise ValueError(f"Unknown dataset {dataset!r}. Supported: {_SUPPORTED}")


def get_inpainting_dataloaders(
    dataset: str,
    batch_size: int,
    data_dir: str | Path,
    mask_generator: MaskGenerator | None = None,
    *,
    image_size: int | None = None,
    mask_type: MaskType | str | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    download: bool = True,
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
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    if key == "fashionmnist":
        return get_fashion_mnist_inpainting_dataloaders(
            batch_size,
            data_dir,
            mask_generator,
            mask_type=mask_type,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    if key == "celeba":
        size = 64 if image_size is None else image_size
        return get_celeba_inpainting_dataloaders(
            batch_size,
            data_dir,
            mask_generator,
            image_size=size,
            mask_type=mask_type,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            download=download,
        )
    raise ValueError(f"Unknown dataset {dataset!r}. Supported: {_SUPPORTED}")


def get_inpainting_dataloaders_from_config(
    config: dict[str, Any],
    mask_generator: MaskGenerator,
    *,
    mask_type: MaskType | str | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build loaders using ``config['dataset']``, ``batch_size``, ``data_dir``."""
    workers = (
        int(num_workers)
        if num_workers is not None
        else int(config.get("num_workers", 0))
    )
    pin = (
        bool(pin_memory)
        if pin_memory is not None
        else bool(config.get("pin_memory", False))
    )
    # Curriculum mutates MaskGenerator.center_ratio each epoch. Persistent
    # workers keep a stale copy, so force respawn unless explicitly overridden.
    if "persistent_workers" in config:
        persist: bool | None = bool(config["persistent_workers"])
    elif config.get("center_ratio_schedule"):
        persist = False
    else:
        persist = None

    return get_inpainting_dataloaders(
        str(config.get("dataset", "MNIST")),
        batch_size=int(config["batch_size"]),
        data_dir=config["data_dir"],
        mask_generator=mask_generator,
        image_size=int(config.get("image_size", 28)),
        mask_type=mask_type,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persist,
        download=download,
    )
