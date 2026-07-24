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

    ↓
    Predict noise ε (same target as standard DDPM)

Reuses :class:`generative_models.ddpm.UNet` with ``in_channels=2*C+1`` —
the architecture is not copied into this repo.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ConditionedUNet(nn.Module):
    """Thin wrapper: concat conditioning channels, then call DDPM U-Net.

    Parameters
    ----------
    image_channels:
        Number of channels in the image (1 for MNIST).
    **unet_kwargs:
        Forwarded to ``generative_models.ddpm.UNet`` (``base_channels``,
        ``channel_mult``, ``image_size``, …). Do not pass ``in_channels`` or
        ``out_channels`` — those are set from ``image_channels``.
    """

    def __init__(self, image_channels: int = 1, **unet_kwargs: Any) -> None:
        super().__init__()
        if image_channels < 1:
            raise ValueError(f"image_channels must be >= 1, got {image_channels}")
        if "in_channels" in unet_kwargs or "out_channels" in unet_kwargs:
            raise TypeError(
                "Pass image_channels=... instead of in_channels/out_channels; "
                "ConditionedUNet sets those from the inpainting concat layout."
            )

        self.image_channels = image_channels
        # in_channels = x_t + masked_image + mask
        self.in_channels = image_channels + image_channels + 1
        self.out_channels = image_channels

        from generative_models.ddpm import UNet

        self.unet = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            **unet_kwargs,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise ``ε̂`` given noisy image and inpainting context.

        Parameters
        ----------
        x_t:
            Noisy image at timestep ``t``, shape ``(B, C, H, W)``.
        t:
            Integer timesteps, shape ``(B,)``.
        masked_image:
            Known pixels only (``x_0 * mask``), shape ``(B, C, H, W)``.
        mask:
            Binary mask (1 = known, 0 = missing), shape ``(B, 1, H, W)``.

        Returns
        -------
        Predicted noise with shape ``(B, C, H, W)``.
        """
        self._validate_inputs(x_t, t, masked_image, mask)
        cond = torch.cat([x_t, masked_image, mask], dim=1)
        return self.unet(cond, t)

    def _validate_inputs(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if x_t.ndim != 4:
            raise ValueError(f"x_t must be (B, C, H, W), got {tuple(x_t.shape)}")
        b, c, h, w = x_t.shape
        if c != self.image_channels:
            raise ValueError(
                f"x_t channels {c} != image_channels {self.image_channels}"
            )
        if masked_image.shape != x_t.shape:
            raise ValueError(
                f"masked_image shape {tuple(masked_image.shape)} must match "
                f"x_t {tuple(x_t.shape)}"
            )
        if mask.shape != (b, 1, h, w):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must be {(b, 1, h, w)}"
            )
        if t.shape != (b,):
            raise ValueError(
                f"t shape {tuple(t.shape)} must be ({b},) matching batch size"
            )


def build_conditioned_unet_from_config(config: dict[str, Any]) -> ConditionedUNet:
    """Build :class:`ConditionedUNet` from a YAML config dict."""
    channel_mult = config.get("channel_mult", [1, 2, 4])
    attention = config.get("attention_resolutions", [7])
    return ConditionedUNet(
        image_channels=int(config.get("image_channels", 1)),
        base_channels=int(config.get("base_channels", 64)),
        channel_mult=tuple(channel_mult),
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        attention_resolutions=tuple(attention),
        dropout=float(config.get("dropout", 0.1)),
        image_size=int(config.get("image_size", 28)),
    )
