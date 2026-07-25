"""Datasets that return (original, masked_image, mask) triples."""

from image_inpainting.datasets.factory import (
    get_base_dataset,
    get_inpainting_dataloaders,
    get_inpainting_dataloaders_from_config,
    normalize_dataset_name,
)
from image_inpainting.datasets.fashion_mnist import (
    get_fashion_mnist_dataset,
    get_fashion_mnist_inpainting_dataloaders,
    get_fashion_mnist_inpainting_datasets,
)
from image_inpainting.datasets.inpainting import InpaintingDataset, apply_mask
from image_inpainting.datasets.mnist import (
    get_mnist_dataset,
    get_mnist_inpainting_dataloaders,
    get_mnist_inpainting_datasets,
)

__all__ = [
    "InpaintingDataset",
    "apply_mask",
    "get_base_dataset",
    "get_fashion_mnist_dataset",
    "get_fashion_mnist_inpainting_dataloaders",
    "get_fashion_mnist_inpainting_datasets",
    "get_inpainting_dataloaders",
    "get_inpainting_dataloaders_from_config",
    "get_mnist_dataset",
    "get_mnist_inpainting_dataloaders",
    "get_mnist_inpainting_datasets",
    "normalize_dataset_name",
]
