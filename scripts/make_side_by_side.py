#!/usr/bin/env python
"""Compose a labeled side-by-side figure from two existing grids.

Panels are scaled (not cropped) to a common height, and each axis gets a width
proportional to its image aspect ratio, so no letterboxing whitespace appears.

Example (CelebA truncated T vs full T):

    python scripts/make_side_by_side.py \\
      --left  path/to/sunglasses_t250.png \\
      --right path/to/conditioned_fullT.png \\
      --left-title  "Truncated T=250 (unsafe)" \\
      --right-title "Full T=1000" \\
      --out docs/assets/celeba_t250_vs_t1000.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side figure with clear captions")
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--left-title", type=str, required=True)
    p.add_argument("--right-title", type=str, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--crop-top",
        type=float,
        default=0.0,
        help="Fraction of height to crop from the top of BOTH images "
        "(removes old suptitles). Try 0.04-0.08; 0 keeps the full image.",
    )
    p.add_argument(
        "--width",
        type=float,
        default=12.0,
        help="Figure width in inches.",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _load(path: Path, crop_top: float) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if crop_top > 0:
        cut = int(round(img.height * crop_top))
        img = img.crop((0, cut, img.width, img.height))
    return img


def main() -> None:
    args = parse_args()
    left = _load(args.left, args.crop_top)
    right = _load(args.right, args.crop_top)

    # Scale to a common height so neither panel loses content.
    height = max(left.height, right.height)
    left = left.resize((round(left.width * height / left.height), height), Image.LANCZOS)
    right = right.resize(
        (round(right.width * height / right.height), height), Image.LANCZOS
    )

    total_width = left.width + right.width
    fig_height = args.width * height / total_width

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(args.width, fig_height),
        gridspec_kw={"width_ratios": [left.width, right.width]},
        constrained_layout=True,
    )
    for ax, img, title in zip(
        axes, (left, right), (args.left_title, args.right_title), strict=True
    ):
        ax.imshow(np.asarray(img))
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.out}  ({left.width}+{right.width} x {height} px panels)")


if __name__ == "__main__":
    main()
