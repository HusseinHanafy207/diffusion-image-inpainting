"""Inpainting metrics: PSNR, SSIM, (optional) LPIPS.

Compare:
    input (masked) → output (inpainted) → ground truth (original)
"""

from __future__ import annotations

from typing import Any

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio (higher is better)."""
    raise NotImplementedError("Phase 6: implement PSNR")


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Structural similarity index (higher is better)."""
    raise NotImplementedError("Phase 6: implement SSIM")


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    include_lpips: bool = False,
) -> dict[str, Any]:
    """Aggregate metrics; optionally restrict to the missing region via ``mask``."""
    raise NotImplementedError("Phase 6: implement compute_metrics")
