"""Display helpers for grayscale and RGB image tensors."""

from __future__ import annotations

import numpy as np
import torch


def tensor_to_display(image: torch.Tensor) -> np.ndarray:
    """Convert ``(C, H, W)`` in ``[0, 1]`` to a matplotlib-friendly array.

    Grayscale → ``(H, W)``; RGB → ``(H, W, 3)``.
    """
    if image.ndim != 3:
        raise ValueError(f"expected (C, H, W), got {tuple(image.shape)}")
    x = image.detach().float().cpu().clamp(0.0, 1.0)
    if x.shape[0] == 1:
        return x[0].numpy()
    if x.shape[0] == 3:
        return x.permute(1, 2, 0).numpy()
    raise ValueError(f"unsupported channel count {x.shape[0]} (want 1 or 3)")


def imshow_tensor(ax: object, image: torch.Tensor) -> None:
    """Show a ``(C, H, W)`` tensor on a matplotlib axis."""
    arr = tensor_to_display(image)
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(arr, vmin=0.0, vmax=1.0)
