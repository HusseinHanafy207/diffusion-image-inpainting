#!/usr/bin/env python
"""Train a mask-conditioned DDPM for image inpainting.

Usage (after install):
    python scripts/train.py --config configs/mnist.yaml
    python scripts/train.py --config configs/mnist.yaml --epochs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train inpainting DDPM")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mnist.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Phase 4: wire MaskGenerator → InpaintingDataset → ConditionedUNet → "
        f"InpaintingTrainer (config={args.config})"
    )


if __name__ == "__main__":
    main()
