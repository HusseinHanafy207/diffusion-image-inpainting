"""Inpainting trainer.

Training is almost identical to standard DDPM:

    original → mask → damaged image → forward diffuse → UNet → MSE(ε̂, ε)

Differences vs ``generative_models.trainers.DDPMTrainer``:
    - each batch draws a fresh mask
    - U-Net sees ``(x_t, masked_image, mask)``
    - loss is still noise-prediction MSE (reuse ``DDPMLoss``)
"""

from __future__ import annotations

from typing import Any


class InpaintingTrainer:
    """Train a conditioned U-Net for inpainting.

    Parameters
    ----------
    model:
        :class:`~image_inpainting.models.ConditionedUNet`.
    scheduler:
        ``generative_models.ddpm.NoiseScheduler``.
    optimizer:
        Torch optimizer.
    device:
        Training device.
    """

    def __init__(
        self,
        model: Any,
        scheduler: Any,
        optimizer: Any,
        device: Any,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader: Any) -> dict[str, float]:
        raise NotImplementedError("Phase 4: implement train_epoch")

    def validate(self, dataloader: Any) -> dict[str, float]:
        raise NotImplementedError("Phase 4: implement validate")

    def fit(self, train_loader: Any, val_loader: Any | None = None, epochs: int = 1) -> None:
        raise NotImplementedError("Phase 4: implement fit")
