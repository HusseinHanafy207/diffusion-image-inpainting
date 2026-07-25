#!/usr/bin/env python
"""Visualize mask types and damaged images via InpaintingDataset.

Usage:
    python scripts/visualize_masks.py --config configs/mnist.yaml
    python scripts/visualize_masks.py --config configs/fashion_mnist.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset

from image_inpainting.datasets import InpaintingDataset, get_base_dataset
from image_inpainting.masks import MaskGenerator, MaskType
from image_inpainting.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize inpainting masks")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mnist.yaml"),
        help="YAML config (selects dataset + figure_dir defaults)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--center-ratio",
        type=float,
        default=None,
        help="Center-hole side length as a fraction of image size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.manual_seed(args.seed)

    dataset_name = str(config.get("dataset", "MNIST"))
    data_dir = args.data_dir or Path(config.get("data_dir", "data/raw"))
    image_size = int(args.image_size or config.get("image_size", 28))
    center_ratio = float(
        args.center_ratio
        if args.center_ratio is not None
        else config.get("center_ratio", 0.4)
    )
    out_dir = args.out_dir or Path(config.get("figure_dir", "outputs/figures"))

    base = get_base_dataset(dataset_name, data_dir, train=True)
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
        gen = MaskGenerator(image_size=image_size, center_ratio=center_ratio)
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

    slug = dataset_name.lower().replace(" ", "_").replace("-", "_")
    fig.suptitle(
        f"InpaintingDataset — {dataset_name} (original → mask → damaged)",
        fontsize=13,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mask_types_{slug}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
