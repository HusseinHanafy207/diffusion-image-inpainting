#!/usr/bin/env python
"""Train a mask-conditioned DDPM for image inpainting.

Usage:
    python scripts/train.py --config configs/mnist.yaml --epochs 1
    python scripts/train.py --config configs/mnist.yaml --epochs 50
    python scripts/train.py --resume outputs/checkpoints/latest.pt --epochs 50
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from generative_models.ddpm import NoiseScheduler
from generative_models.losses import DDPMLoss

from image_inpainting.datasets import get_inpainting_dataloaders_from_config
from image_inpainting.masks import build_mask_generator_from_config
from image_inpainting.models import build_conditioned_unet_from_config
from image_inpainting.trainers import InpaintingTrainer
from image_inpainting.utils import load_config


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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default=None,
        help="Optional fixed mask type (center|rectangle|brush|holes). "
        "Default: sample from config mask_types each item.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override checkpoint_dir (use a Drive path on Colab).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Override log_dir (use a Drive path on Colab).",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=None,
        help="Override sample_dir (use a Drive path on Colab).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data_dir for dataset download/cache.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: config num_workers, else 2 on CUDA / 0 on CPU).",
    )
    return parser.parse_args()


def print_sanity_checks(train_metrics: dict[str, float], config: dict) -> None:
    print("\nPhase 4 sanity checks")
    print("-" * 24)

    finite = math.isfinite(train_metrics["loss"])
    print(f"[{'OK' if finite else 'FAIL'}] Loss is finite (no NaN/Inf)")

    batch_improved = train_metrics["last_batch_loss"] <= train_metrics["first_batch_loss"]
    print(
        f"[{'OK' if batch_improved else 'WARN'}] Batch loss trend: "
        f"{train_metrics['first_batch_loss']:.4f} -> {train_metrics['last_batch_loss']:.4f}"
    )

    checkpoint_path = Path(config["checkpoint_dir"]) / "latest.pt"
    print(f"[{'OK' if checkpoint_path.exists() else 'FAIL'}] Checkpoint saved")

    train_csv = Path(config["log_dir"]) / config.get(
        "train_metrics_file", "train_metrics.csv"
    )
    print(f"[{'OK' if train_csv.exists() else 'FAIL'}] Train metrics CSV written")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.log_dir is not None:
        config["log_dir"] = str(args.log_dir)
    if args.sample_dir is not None:
        config["sample_dir"] = str(args.sample_dir)
    if args.data_dir is not None:
        config["data_dir"] = str(args.data_dir)

    # Resolve device early so pin_memory / worker defaults match the run.
    from generative_models.utils.device import get_device

    if args.device is None or args.device == "auto":
        device = get_device()
        config["device"] = "auto"
    else:
        device = torch.device(args.device)
        config["device"] = args.device

    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    elif "num_workers" not in config:
        config["num_workers"] = 2 if device.type == "cuda" else 0

    if "pin_memory" not in config:
        config["pin_memory"] = device.type == "cuda"

    if args.resume is None and config.get("seed") is not None:
        torch.manual_seed(int(config["seed"]))

    mask_generator = build_mask_generator_from_config(config)
    train_loader, val_loader = get_inpainting_dataloaders_from_config(
        config,
        mask_generator,
        mask_type=args.mask_type,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
    )

    model = build_conditioned_unet_from_config(config)
    scheduler = NoiseScheduler(
        num_timesteps=int(config["num_timesteps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = DDPMLoss(reduction="mean")

    trainer = InpaintingTrainer(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        config["start_epoch"] = checkpoint["epoch"]
        if config["epochs"] <= checkpoint["epoch"]:
            raise ValueError(
                f"--epochs {config['epochs']} must be greater than resumed epoch "
                f"{checkpoint['epoch']}."
            )
        # Checkpoint Adam state keeps the old LR; apply the config rate so LR
        # experiments (e.g. Phase 2) actually take effect.
        resume_lr = float(config["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = resume_lr
        print(
            f"Resumed from epoch {checkpoint['epoch']}, "
            f"training to epoch {config['epochs']} | "
            f"optimizer lr set to {resume_lr}"
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Dataset: {config.get('dataset', 'MNIST')}")
    print(f"Device: {device}")
    print(f"ConditionedUNet parameters: {n_params:,}")
    type_mix = ", ".join(
        f"{t.value}={p:.2f}"
        for t, p in zip(mask_generator.mask_types, mask_generator.mask_type_probs.tolist())
    )
    print(
        f"Mask mix: {type_mix} | center_jitter_ratio="
        f"{mask_generator.center_jitter_ratio}"
    )
    print(
        f"Training for {config['epochs']} epochs | "
        f"batch_size={config['batch_size']} | lr={config['learning_rate']} | "
        f"num_workers={config['num_workers']} | pin_memory={config['pin_memory']}"
    )
    print()

    trainer.train()

    checkpoint = torch.load(
        Path(config["checkpoint_dir"]) / "latest.pt",
        weights_only=False,
    )
    train_metrics = checkpoint["metrics"]
    metrics_file = Path(config["log_dir"]) / config["train_metrics_file"]
    print(f"\nFinished training through epoch {config['epochs']}.")
    print(f"Train metrics CSV: {metrics_file}")
    print(f"Latest checkpoint: {Path(config['checkpoint_dir']) / 'latest.pt'}")

    if config["epochs"] == 1 and args.resume is None:
        print_sanity_checks(train_metrics, config)


if __name__ == "__main__":
    main()
