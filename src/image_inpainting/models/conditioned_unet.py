"""U-Net conditioned on the masked image and binary mask.

The base DDPM U-Net in ``generative_models.ddpm`` takes ``x_t`` only
(``in_channels=1`` for MNIST). For inpainting we concatenate:

    [x_t | masked_image | mask]  →  in_channels = C + C + 1

Conceptually:

    UNet input
    ----------
    noisy image     x_t
    known pixels    x_0 ⊙ m
    mask            m

and the network predicts noise (or the missing region indirectly via ε).

Implementation plan:
    Reuse :class:`generative_models.ddpm.UNet` with ``in_channels=2*C+1``
    rather than copying the architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionedUNet(nn.Module):
    """Thin wrapper: concat conditioning channels, then call DDPM U-Net.

    Parameters
    ----------
    image_channels:
        Number of channels in the image (1 for MNIST).
    **unet_kwargs:
        Forwarded to ``generative_models.ddpm.UNet`` (base_channels, etc.).
    """

    def __init__(self, image_channels: int = 1, **unet_kwargs) -> None:
        super().__init__()
        self.image_channels = image_channels
        # in_channels = x_t + masked_image + mask
        in_channels = image_channels + image_channels + 1
        # Lazy import so installing this package without generative-models
        # still allows inspecting the scaffold.
        from generative_models.ddpm import UNet

        self.unet = UNet(
            in_channels=in_channels,
            out_channels=image_channels,
            **unet_kwargs,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise given noisy image and inpainting context."""
        cond = torch.cat([x_t, masked_image, mask], dim=1)
        return self.unet(cond, t)
