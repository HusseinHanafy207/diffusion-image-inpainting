#!/usr/bin/env python
"""Run reverse-diffusion inpainting on masked MNIST images.

Usage:
    python scripts/inpaint.py --checkpoint outputs/checkpoints/latest.pt --seed 42
    python scripts/inpaint.py --checkpoint outputs/checkpoints/epoch_020.pt --mask-type center
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from generative_models.utils.device import get_device
from torch.utils.data import Subset

from image_inpainting.datasets import InpaintingDataset, get_mnist_dataset
from image_inpainting.diffusion import inpaint, load_inpainting_checkpoint
from image_inpainting.masks import MaskGenerator, MaskType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inpaint with conditioned DDPM")
    parser.add_argument("--config", type=Path, default=Path("configs/mnist.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mask-type",
        type=str,
        default="center",
        choices=[m.value for m in MaskType],
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/samples"))
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override MNIST data_dir from config",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional fewer reverse steps for a quick preview (default: full T)",
    )
    parser.add_argument(
        "--jump-length",
        type=int,
        default=10,
        help="RePaint forward jump length j (paper default 10)",
    )
    parser.add_argument(
        "--jump-n-sample",
        type=int,
        default=10,
        help="RePaint resample count r (paper default 10). Use 1 to disable.",
    )
    return parser.parse_args()


def save_comparison_grid(
    originals: torch.Tensor,
    masked: torch.Tensor,
    inpainted: torch.Tensor,
    output_path: Path,
    title: str,
) -> Path:
    """Save rows: original | damaged | inpainted for each sample."""
    n = originals.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(6.0, 2.0 * n), squeeze=False)
    col_titles = ["original", "damaged", "inpainted"]

    for i in range(n):
        panels = (
            originals[i, 0].cpu().numpy(),
            masked[i, 0].cpu().numpy(),
            inpainted[i, 0].cpu().numpy(),
        )
        for k, panel in enumerate(panels):
            ax = axes[i][k]
            ax.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[k], fontsize=11)
            if k == 0:
                ax.set_ylabel(f"#{i}", fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    model, scheduler, config, checkpoint = load_inpainting_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        config_path=args.config,
    )
    epoch = checkpoint.get("epoch", "?")
    data_dir = args.data_dir or config.get("data_dir", "data/raw")
    image_size = int(config.get("image_size", 28))

    effective_t = args.timesteps if args.timesteps is not None else scheduler.num_timesteps
    print(f"Loaded checkpoint from epoch {epoch}")
    print(
        f"Device: {device} | T={effective_t} | mask={args.mask_type} | "
        f"jump_length={args.jump_length} | jump_n_sample={args.jump_n_sample}"
    )

    base = get_mnist_dataset(data_dir, train=False)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(base), generator=generator)[: args.num_samples].tolist()
    subset = Subset(base, indices)
    ds = InpaintingDataset(
        subset,
        MaskGenerator(image_size=image_size),
        mask_type=args.mask_type,
    )

    originals = []
    masked_images = []
    masks = []
    for i in range(args.num_samples):
        original, masked, mask = ds[i]
        originals.append(original)
        masked_images.append(masked)
        masks.append(mask)

    originals_b = torch.stack(originals, dim=0).to(device)
    masked_b = torch.stack(masked_images, dim=0).to(device)
    masks_b = torch.stack(masks, dim=0).to(device)

    print(f"Running reverse inpainting on {args.num_samples} images…")
    result = inpaint(
        model,
        scheduler,
        masked_b,
        masks_b,
        original=originals_b,
        num_timesteps=args.timesteps,
        jump_length=args.jump_length,
        jump_n_sample=args.jump_n_sample,
        show_progress=True,
    )

    # Known pixels must match the original exactly.
    known_err = ((result - originals_b) * masks_b).abs().max().item()
    print(f"Max |error| on known pixels: {known_err:.2e}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    epoch_tag = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
    resample_tag = (
        f"_j{args.jump_length}r{args.jump_n_sample}"
        if args.jump_n_sample > 1
        else ""
    )
    out_path = (
        args.out_dir / f"inpaint_{args.mask_type}_epoch_{epoch_tag}{resample_tag}.png"
    )
    title = f"Inpainting ({args.mask_type}, epoch {epoch}"
    if args.jump_n_sample > 1:
        title += f", RePaint j={args.jump_length} r={args.jump_n_sample}"
    title += ")"
    save_comparison_grid(
        originals_b,
        masked_b,
        result,
        out_path,
        title=title,
    )
    print(f"Saved comparison grid to {out_path}")


if __name__ == "__main__":
    main()
