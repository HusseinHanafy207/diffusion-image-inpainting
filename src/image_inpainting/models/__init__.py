"""Conditioned models for inpainting (built on generative_models.ddpm)."""

from image_inpainting.models.conditioned_unet import ConditionedUNet

__all__ = ["ConditionedUNet"]
