#!/usr/bin/env python
"""Run reverse-diffusion inpainting on masked images.

Usage:
    python scripts/inpaint.py --checkpoint outputs/checkpoints/latest.pt --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inpaint with conditioned DDPM")
    parser.add_argument("--config", type=Path, default=Path("configs/mnist.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask-type", type=str, default="center")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/samples"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Phase 5: load checkpoint, apply masks, call image_inpainting.diffusion.inpaint "
        f"(checkpoint={args.checkpoint})"
    )


if __name__ == "__main__":
    main()
