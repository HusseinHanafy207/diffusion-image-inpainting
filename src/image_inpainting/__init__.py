"""Image inpainting with mask-conditioned DDPM.

This package builds on ``generative_models.ddpm`` (sibling repo) and adds
masking, conditioning, training, and RePaint-style inference. It does **not**
reimplement the noise schedule, U-Net backbone, or forward diffusion.
"""

__version__ = "0.1.0"
