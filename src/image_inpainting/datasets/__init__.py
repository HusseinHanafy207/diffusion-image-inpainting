"""Datasets that return (original, masked_image, mask) triples."""

from image_inpainting.datasets.inpainting import InpaintingDataset, apply_mask

__all__ = ["InpaintingDataset", "apply_mask"]
