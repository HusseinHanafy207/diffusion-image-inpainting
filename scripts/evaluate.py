#!/usr/bin/env python
"""Evaluate inpainting quality (PSNR / SSIM).

Compares damaged input and inpainted output against ground truth.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/latest.pt
    python scripts/evaluate.py --checkpoint outputs/checkpoints/epoch_040.pt \\
        --mask-type center --max-samples 32 --jump-length 10 --jump-n-sample 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from generative_models.utils.device import get_device
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from image_inpainting.datasets import InpaintingDataset, get_base_dataset
from image_inpainting.diffusion import inpaint, load_inpainting_checkpoint
from image_inpainting.evaluation import compute_metrics
from image_inpainting.masks import MaskGenerator, MaskType
from image_inpainting.utils import imshow_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inpainting PSNR / SSIM")
    parser.add_argument("--config", type=Path, default=Path("configs/mnist.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mask-type",
        type=str,
        default="center",
        choices=[m.value for m in MaskType],
    )
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override reverse steps (default: full T from checkpoint)",
    )
    parser.add_argument("--jump-length", type=int, default=10)
    parser.add_argument(
        "--jump-n-sample",
        type=int,
        default=5,
        help="RePaint resample count r (1 = noise-matched stitch only)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/eval"),
        help="Directory for metrics CSV/JSON and comparison grid",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=8,
        help="How many samples to include in the visual grid",
    )
    return parser.parse_args()


def _mean_std(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.mean().item()), float(t.std(unbiased=False).item())


def save_comparison_grid(
    originals: torch.Tensor,
    masked: torch.Tensor,
    inpainted: torch.Tensor,
    output_path: Path,
    title: str,
) -> Path:
    """Rows: ground truth | input (damaged) | output (inpainted)."""
    n = originals.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(6.0, 2.0 * n), squeeze=False)
    col_titles = ["ground truth", "input", "output"]

    for i in range(n):
        panels = (originals[i], masked[i], inpainted[i])
        for k, panel in enumerate(panels):
            ax = axes[i][k]
            imshow_tensor(ax, panel)
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

    print(f"Loaded checkpoint from epoch {epoch}")
    print(
        f"Device: {device} | dataset={config.get('dataset', 'MNIST')} | "
        f"mask={args.mask_type} | samples={args.max_samples} | "
        f"jump_length={args.jump_length} | jump_n_sample={args.jump_n_sample}"
    )

    base = get_base_dataset(
        str(config.get("dataset", "MNIST")),
        data_dir,
        train=False,
        image_size=image_size,
    )
    generator = torch.Generator().manual_seed(args.seed)
    n = min(args.max_samples, len(base))
    indices = torch.randperm(len(base), generator=generator)[:n].tolist()
    ds = InpaintingDataset(
        Subset(base, indices),
        MaskGenerator(image_size=image_size),
        mask_type=args.mask_type,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    per_image: list[dict[str, float]] = []
    vis_originals: list[torch.Tensor] = []
    vis_masked: list[torch.Tensor] = []
    vis_inpainted: list[torch.Tensor] = []

    for originals, masked, masks in tqdm(loader, desc="evaluate"):
        originals = originals.to(device)
        masked = masked.to(device)
        masks = masks.to(device)

        preds = inpaint(
            model,
            scheduler,
            masked,
            masks,
            original=originals,
            num_timesteps=args.timesteps,
            jump_length=args.jump_length,
            jump_n_sample=args.jump_n_sample,
            show_progress=False,
        )

        for i in range(originals.shape[0]):
            o = originals[i : i + 1]
            m = masked[i : i + 1]
            p = preds[i : i + 1]
            mk = masks[i : i + 1]

            out_m = compute_metrics(p, o, mask=mk)
            in_m = compute_metrics(m, o, mask=mk)
            per_image.append(
                {
                    "psnr_input": in_m["psnr"],
                    "ssim_input": in_m["ssim"],
                    "psnr_hole_input": in_m["psnr_hole"],
                    "psnr_output": out_m["psnr"],
                    "ssim_output": out_m["ssim"],
                    "psnr_hole_output": out_m["psnr_hole"],
                }
            )

            if len(vis_originals) < args.num_vis:
                vis_originals.append(o[0].cpu())
                vis_masked.append(m[0].cpu())
                vis_inpainted.append(p[0].cpu())

    def col(name: str) -> list[float]:
        return [row[name] for row in per_image]

    summary = {
        "epoch": epoch,
        "dataset": str(config.get("dataset", "MNIST")),
        "mask_type": args.mask_type,
        "num_samples": len(per_image),
        "jump_length": args.jump_length,
        "jump_n_sample": args.jump_n_sample,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "psnr_input_mean": _mean_std(col("psnr_input"))[0],
        "psnr_input_std": _mean_std(col("psnr_input"))[1],
        "ssim_input_mean": _mean_std(col("ssim_input"))[0],
        "ssim_input_std": _mean_std(col("ssim_input"))[1],
        "psnr_output_mean": _mean_std(col("psnr_output"))[0],
        "psnr_output_std": _mean_std(col("psnr_output"))[1],
        "ssim_output_mean": _mean_std(col("ssim_output"))[0],
        "ssim_output_std": _mean_std(col("ssim_output"))[1],
        "psnr_hole_input_mean": _mean_std(col("psnr_hole_input"))[0],
        "psnr_hole_output_mean": _mean_std(col("psnr_hole_output"))[0],
        "psnr_hole_output_std": _mean_std(col("psnr_hole_output"))[1],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    epoch_tag = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
    stem = f"eval_{args.mask_type}_epoch_{epoch_tag}_j{args.jump_length}r{args.jump_n_sample}"

    csv_path = args.out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    json_path = args.out_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    grid_path = args.out_dir / f"{stem}_grid.png"
    save_comparison_grid(
        torch.stack(vis_originals),
        torch.stack(vis_masked),
        torch.stack(vis_inpainted),
        grid_path,
        title=(
            f"Eval ({args.mask_type}, epoch {epoch}, "
            f"RePaint j={args.jump_length} r={args.jump_n_sample})"
        ),
    )

    print("\nPhase 6 - evaluation summary")
    print("-" * 40)
    print(f"Samples:           {summary['num_samples']}")
    print(
        f"Input  PSNR / SSIM: "
        f"{summary['psnr_input_mean']:.2f} / {summary['ssim_input_mean']:.4f}"
    )
    print(
        f"Output PSNR / SSIM: "
        f"{summary['psnr_output_mean']:.2f} +/- {summary['psnr_output_std']:.2f} / "
        f"{summary['ssim_output_mean']:.4f} +/- {summary['ssim_output_std']:.4f}"
    )
    print(
        f"Hole   PSNR (in->out): "
        f"{summary['psnr_hole_input_mean']:.2f} -> {summary['psnr_hole_output_mean']:.2f}"
    )
    print(f"Per-image CSV:     {csv_path}")
    print(f"Summary JSON:      {json_path}")
    print(f"Comparison grid:   {grid_path}")


if __name__ == "__main__":
    main()
