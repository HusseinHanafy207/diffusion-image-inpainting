#!/usr/bin/env python
"""Visualize mask types and damaged MNIST digits.

Usage:
    python scripts/visualize_masks.py --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from image_inpainting.datasets import apply_mask
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


def load_mnist_images(data_dir: Path, num_examples: int, seed: int) -> torch.Tensor:
    dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_examples]
    images = torch.stack([dataset[int(i)][0] for i in indices], dim=0)
    return images


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    images = load_mnist_images(args.data_dir, args.num_examples, args.seed)
    gen = MaskGenerator(image_size=args.image_size, center_ratio=args.center_ratio)
    mask_types = list(MaskType)

    n_rows = len(mask_types)
    n_cols = args.num_examples
    # For each mask type: original | mask | damaged  → 3 panels per example would be wide.
    # Layout: rows = mask types, cols = examples; each cell shows damaged with mask overlay.
    fig, axes = plt.subplots(
        n_rows,
        n_cols * 3,
        figsize=(2.0 * n_cols * 3, 2.0 * n_rows),
        squeeze=False,
    )

    col_titles = ["original", "mask", "damaged"]
    for row, mask_type in enumerate(mask_types):
        masks = gen(batch_size=args.num_examples, mask_type=mask_type)
        damaged = apply_mask(images, masks)

        for col in range(n_cols):
            panels = (
                images[col, 0].numpy(),
                masks[col, 0].numpy(),
                damaged[col, 0].numpy(),
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

    fig.suptitle("MaskGenerator — MNIST (1 = known, 0 = missing)", fontsize=13)
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "mask_types_mnist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
