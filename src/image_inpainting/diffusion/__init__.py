"""Inpainting-specific diffusion helpers (reuse DDPM forward/schedule)."""

from image_inpainting.diffusion.inpaint_sampler import inpaint

__all__ = ["inpaint"]
