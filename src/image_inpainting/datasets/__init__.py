"""Datasets that return (original, masked_image, mask) triples."""

from image_inpainting.datasets.inpainting import InpaintingDataset, apply_mask
from image_inpainting.datasets.mnist import (
    get_mnist_dataset,
    get_mnist_inpainting_dataloaders,
    get_mnist_inpainting_datasets,
)

__all__ = [
    "InpaintingDataset",
    "apply_mask",
    "get_mnist_dataset",
    "get_mnist_inpainting_dataloaders",
    "get_mnist_inpainting_datasets",
]
