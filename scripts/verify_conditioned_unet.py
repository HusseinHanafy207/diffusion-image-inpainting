#!/usr/bin/env python
"""Verify ConditionedUNet shapes, gradients, and channel layout.

Usage:
    python scripts/verify_conditioned_unet.py
"""

from __future__ import annotations

import torch

from generative_models.ddpm import NoiseScheduler, forward_diffuse

from image_inpainting.datasets import get_mnist_inpainting_dataloaders
from image_inpainting.masks import MaskGenerator, MaskType
from image_inpainting.models import ConditionedUNet
from image_inpainting.utils import load_config


def main() -> None:
    config = load_config("configs/mnist.yaml")
    model = ConditionedUNet(
        image_channels=int(config["image_channels"]),
        base_channels=int(config["base_channels"]),
        channel_mult=tuple(config["channel_mult"]),
        num_res_blocks=int(config["num_res_blocks"]),
        attention_resolutions=tuple(config["attention_resolutions"]),
        dropout=float(config["dropout"]),
        image_size=int(config["image_size"]),
    )
    n_params = sum(p.numel() for p in model.parameters())

    print("=== ConditionedUNet ===")
    print(f"parameters:     {n_params:,}")
    print(f"in_channels:   {model.in_channels}  (x_t + masked_image + mask)")
    print(f"out_channels:  {model.out_channels}")

    train_loader, _ = get_mnist_inpainting_dataloaders(
        batch_size=8,
        data_dir=config["data_dir"],
        mask_generator=MaskGenerator(image_size=int(config["image_size"])),
        mask_type=MaskType.CENTER,
    )
    x_0, masked_image, mask = next(iter(train_loader))
    scheduler = NoiseScheduler(
        num_timesteps=int(config["num_timesteps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    )
    x_t, t, noise = forward_diffuse(scheduler, x_0)

    pred = model(x_t, t, masked_image, mask)
    print(f"x_0 shape:     {tuple(x_0.shape)}")
    print(f"masked shape:  {tuple(masked_image.shape)}")
    print(f"mask shape:    {tuple(mask.shape)}")
    print(f"x_t shape:     {tuple(x_t.shape)}")
    print(f"concat layout: (B, {model.in_channels}, H, W)")
    print(f"eps_hat shape: {tuple(pred.shape)}")

    loss = torch.mean((pred - noise) ** 2)
    loss.backward()
    print(f"mse(eps_hat, eps) = {loss.item():.4f}")
    print(f"grad on conv_in: {model.unet.conv_in.weight.grad is not None}")
    print(f"conv_in weight:  {tuple(model.unet.conv_in.weight.shape)}")

    model.eval()
    with torch.no_grad():
        y_early = model(x_t[:1], torch.tensor([0]), masked_image[:1], mask[:1])
        y_late = model(x_t[:1], torch.tensor([999]), masked_image[:1], mask[:1])
        diff = (y_early - y_late).abs().mean().item()
    print(f"mean |f(..., t=0) - f(..., t=999)| = {diff:.4f}  (should be > 0)")

    ok = (
        pred.shape == x_0.shape
        and model.in_channels == 3
        and model.unet.conv_in.weight.shape[1] == 3
        and diff > 0
        and model.unet.conv_in.weight.grad is not None
    )
    print("\nConditionedUNet looks correct." if ok else "\nConditionedUNet mismatch!")


if __name__ == "__main__":
    main()
