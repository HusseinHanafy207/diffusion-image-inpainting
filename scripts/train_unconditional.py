#!/usr/bin/env python
"""Train an unconditional DDPM (no masks) for RePaint inpainting at inference.

Unlike ``scripts/train.py`` (mask-conditioned), this trains a standard face
generator on clean CelebA images only. Inpainting constraints are applied
later via RePaint in ``scripts/inpaint.py --unconditional``.

Usage:
    python scripts/train_unconditional.py --config configs/celeba_unconditional.yaml --epochs 1
    python scripts/train_unconditional.py --config configs/celeba_unconditional.yaml --epochs 40
    python scripts/train_unconditional.py --resume outputs/celeba_uncond/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from generative_models.losses import DDPMLoss
from generative_models.trainers import DDPMTrainer
from generative_models.utils.device import get_device
from torch.utils.data import DataLoader

from image_inpainting.datasets import get_base_dataset
from image_inpainting.datasets.loader_utils import build_dataloader_kwargs
from image_inpainting.models.unconditional import build_unconditional_ddpm_from_config
from image_inpainting.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unconditional DDPM")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/celeba_unconditional.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    parser.add_argument("--device", type=str, default=None, help="auto | cpu | cuda")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def print_sanity_checks(train_metrics: dict[str, float], config: dict) -> None:
    print("\nUnconditional DDPM sanity checks")
    print("-" * 32)
    finite = math.isfinite(train_metrics["loss"])
    print(f"[{'OK' if finite else 'FAIL'}] Loss is finite (no NaN/Inf)")
    batch_improved = train_metrics["last_batch_loss"] <= train_metrics["first_batch_loss"]
    print(
        f"[{'OK' if batch_improved else 'WARN'}] Batch loss trend: "
        f"{train_metrics['first_batch_loss']:.4f} -> {train_metrics['last_batch_loss']:.4f}"
    )
    checkpoint_path = Path(config["checkpoint_dir"]) / "latest.pt"
    print(f"[{'OK' if checkpoint_path.exists() else 'FAIL'}] Checkpoint saved")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["training_mode"] = "unconditional"

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

    image_size = int(config.get("image_size", 64))
    train_base = get_base_dataset(
        str(config.get("dataset", "CelebA")),
        config["data_dir"],
        train=True,
        image_size=image_size,
    )
    val_base = get_base_dataset(
        str(config.get("dataset", "CelebA")),
        config["data_dir"],
        train=False,
        image_size=image_size,
    )
    loader_kwargs = build_dataloader_kwargs(
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
    )
    train_loader = DataLoader(
        train_base,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_base,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )

    model = build_unconditional_ddpm_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = DDPMLoss(reduction="mean")

    trainer = DDPMTrainer(
        model=model,
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
        resume_lr = float(config["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = resume_lr
        print(
            f"Resumed from epoch {checkpoint['epoch']}, "
            f"training to epoch {config['epochs']} | "
            f"optimizer lr set to {resume_lr}"
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Dataset: {config.get('dataset', 'CelebA')} (clean images only — no masks)")
    print(f"Device: {device}")
    print(f"Unconditional DDPM parameters: {n_params:,}")
    print(
        f"image_channels={config.get('image_channels')} | "
        f"image_size={image_size} | T={config['num_timesteps']}"
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
