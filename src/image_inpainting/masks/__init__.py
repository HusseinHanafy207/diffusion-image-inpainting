"""Mask generation for inpainting experiments."""

from image_inpainting.masks.config import build_mask_generator_from_config
from image_inpainting.masks.generator import MaskGenerator, MaskType

__all__ = ["MaskGenerator", "MaskType", "build_mask_generator_from_config"]
