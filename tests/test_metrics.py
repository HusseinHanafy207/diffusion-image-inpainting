"""Tests for PSNR / SSIM evaluation metrics."""

from __future__ import annotations

import math

import torch

from image_inpainting.evaluation import compute_metrics, psnr, ssim
from image_inpainting.evaluation.metrics import psnr_masked


def test_psnr_identical_is_infinite() -> None:
    x = torch.rand(2, 1, 28, 28)
    assert math.isinf(psnr(x, x))


def test_psnr_decreases_with_noise() -> None:
    torch.manual_seed(0)
    target = torch.rand(1, 1, 28, 28)
    mild = (target + 0.01 * torch.randn_like(target)).clamp(0, 1)
    heavy = (target + 0.2 * torch.randn_like(target)).clamp(0, 1)
    assert psnr(mild, target) > psnr(heavy, target)


def test_ssim_identical_near_one() -> None:
    x = torch.rand(2, 1, 28, 28)
    assert ssim(x, x) > 0.99


def test_compute_metrics_with_hole() -> None:
    torch.manual_seed(1)
    target = torch.rand(1, 1, 28, 28)
    mask = torch.ones(1, 1, 28, 28)
    mask[:, :, 8:20, 8:20] = 0.0
    pred = target.clone()
    pred = pred * mask  # hole is wrong (zeros)

    metrics = compute_metrics(pred, target, mask=mask)
    assert "psnr" in metrics and "ssim" in metrics and "psnr_hole" in metrics
    assert metrics["psnr_hole"] < metrics["psnr"]
    assert psnr_masked(target, target, mask) == float("inf")
