#!/usr/bin/env python
"""Evaluate inpainting quality (PSNR / SSIM / optional LPIPS).

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inpainting metrics")
    parser.add_argument("--config", type=Path, default=Path("configs/mnist.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask-type", type=str, default="center")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Phase 6: inpaint held-out set and report PSNR/SSIM "
        f"(checkpoint={args.checkpoint})"
    )


if __name__ == "__main__":
    main()
