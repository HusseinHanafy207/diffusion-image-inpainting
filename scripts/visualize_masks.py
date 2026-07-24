#!/usr/bin/env python
"""Visualize mask types and damaged MNIST digits via InpaintingDataset.

Usage:
    python scripts/visualize_masks.py --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset

from image_inpainting.datasets import (
    InpaintingDataset,
    get_mnist_dataset,
)
from image_inpainting.masks import MaskGenerator, MaskType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize inpainting masks")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--center-ratio",
        type=float,
        default=0.4,
        help="Center-hole side length as a fraction of image size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    base = get_mnist_dataset(args.data_dir, train=True)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(base), generator=generator)[: args.num_examples].tolist()
    subset = Subset(base, indices)

    mask_types = list(MaskType)
    n_rows = len(mask_types)
    n_cols = args.num_examples
    fig, axes = plt.subplots(
        n_rows,
        n_cols * 3,
        figsize=(2.0 * n_cols * 3, 2.0 * n_rows),
        squeeze=False,
    )

    col_titles = ["original", "mask", "damaged"]
    for row, mask_type in enumerate(mask_types):
        gen = MaskGenerator(image_size=args.image_size, center_ratio=args.center_ratio)
        ds = InpaintingDataset(subset, gen, mask_type=mask_type)

        for col in range(n_cols):
            original, masked, mask = ds[col]
            panels = (
                original[0].numpy(),
                mask[0].numpy(),
                masked[0].numpy(),
            )
            for k, panel in enumerate(panels):
                ax = axes[row][col * 3 + k]
                ax.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0 and k == 0:
                    ax.set_ylabel(mask_type.value, fontsize=11)
                if row == 0:
                    ax.set_title(f"{col_titles[k]}\n#{col}", fontsize=9)

    fig.suptitle(
        "InpaintingDataset — MNIST (original → mask → damaged)",
        fontsize=13,
    )
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "mask_types_mnist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
