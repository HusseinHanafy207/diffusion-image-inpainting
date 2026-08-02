"""Inpainting metrics: PSNR, SSIM, and optional LPIPS.

Compare:
    input (masked) → output (inpainted) → ground truth (original)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must match target {tuple(target.shape)}"
        )
    if pred.ndim not in (3, 4):
        raise ValueError(f"expected (C,H,W) or (B,C,H,W), got {tuple(pred.shape)}")


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio over the full tensor (higher is better)."""
    _validate_pair(pred, target)
    if data_range <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")

    pred_f = pred.detach().float()
    target_f = target.detach().float()
    mse = torch.mean((pred_f - target_f) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def psnr_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    data_range: float = 1.0,
    missing_value: float = 0.0,
) -> float:
    """PSNR only on pixels where ``mask == missing_value`` (the hole)."""
    _validate_pair(pred, target)
    hole = mask == missing_value
    if hole.shape[-2:] != pred.shape[-2:]:
        raise ValueError(
            f"mask spatial shape {tuple(hole.shape[-2:])} incompatible with "
            f"pred {tuple(pred.shape[-2:])}"
        )
    if not hole.any():
        raise ValueError("mask has no missing pixels to evaluate")

    # Broadcast mask over channels if needed.
    while hole.ndim < pred.ndim:
        hole = hole.unsqueeze(0 if hole.ndim == pred.ndim - 1 else 1)
    if hole.shape[1] == 1 and pred.shape[1] > 1:
        hole = hole.expand_as(pred)
    elif hole.shape != pred.shape:
        hole = hole.expand_as(pred)

    pred_f = pred.detach().float()[hole]
    target_f = target.detach().float()[hole]
    mse = torch.mean((pred_f - target_f) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Mean structural similarity over a batch (higher is better, in [-1, 1])."""
    _validate_pair(pred, target)
    if data_range <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")

    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    scores: list[float] = []
    for i in range(pred.shape[0]):
        p = pred[i].detach().float().cpu().numpy()
        t = target[i].detach().float().cpu().numpy()
        # skimage expects (H, W) or (H, W, C)
        if p.shape[0] == 1:
            p_img, t_img = p[0], t[0]
            score = structural_similarity(t_img, p_img, data_range=data_range)
        else:
            p_img = np.transpose(p, (1, 2, 0))
            t_img = np.transpose(t, (1, 2, 0))
            score = structural_similarity(
                t_img,
                p_img,
                data_range=data_range,
                channel_axis=-1,
            )
        scores.append(float(score))
    return float(np.mean(scores))


def _to_lpips_input(x: torch.Tensor, data_range: float) -> torch.Tensor:
    """Map ``[0, data_range]`` images to LPIPS ``[-1, 1]`` RGB input."""
    x = x.detach().float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if data_range <= 0:
        raise ValueError(f"data_range must be > 0, got {data_range}")
    x01 = (x / data_range).clamp(0.0, 1.0)
    if x01.shape[1] == 1:
        x01 = x01.repeat(1, 3, 1, 1)
    elif x01.shape[1] != 3:
        raise ValueError(f"LPIPS expects 1 or 3 channels, got {x01.shape[1]}")
    return x01 * 2.0 - 1.0


@torch.no_grad()
def lpips_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_fn: Any,
    *,
    data_range: float = 1.0,
) -> float:
    """Mean LPIPS distance (lower is better). ``lpips_fn`` is an ``lpips.LPIPS`` module."""
    _validate_pair(pred, target)
    d = lpips_fn(
        _to_lpips_input(pred, data_range),
        _to_lpips_input(target, data_range),
    )
    return float(d.mean().item())


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    data_range: float = 1.0,
    lpips_fn: Any | None = None,
) -> dict[str, Any]:
    """Aggregate PSNR / SSIM; optionally hole-only PSNR and LPIPS.

    Parameters
    ----------
    pred, target:
        Images in ``[0, data_range]``, shape ``(B, C, H, W)`` or ``(C, H, W)``.
    mask:
        If set (1 = known, 0 = missing), also report ``psnr_hole``.
    lpips_fn:
        Optional ``lpips.LPIPS`` module; if set, also report ``lpips``.
    """
    metrics: dict[str, Any] = {
        "psnr": psnr(pred, target, data_range=data_range),
        "ssim": ssim(pred, target, data_range=data_range),
    }
    if mask is not None:
        metrics["psnr_hole"] = psnr_masked(
            pred, target, mask, data_range=data_range, missing_value=0.0
        )
    if lpips_fn is not None:
        metrics["lpips"] = lpips_distance(
            pred, target, lpips_fn, data_range=data_range
        )
    return metrics
