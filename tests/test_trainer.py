"""Smoke tests for InpaintingTrainer (tiny model + tiny data subset)."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from generative_models.ddpm import NoiseScheduler
from generative_models.losses import DDPMLoss

from image_inpainting.datasets import InpaintingDataset, get_mnist_dataset
from image_inpainting.masks import MaskGenerator, MaskType
from image_inpainting.models import ConditionedUNet
from image_inpainting.trainers import InpaintingTrainer
from image_inpainting.utils import load_config


def test_inpainting_trainer_runs_one_epoch(tmp_path: Path) -> None:
    config = load_config("configs/mnist.yaml")
    config.update(
        {
            "epochs": 1,
            "device": "cpu",
            "batch_size": 8,
            "base_channels": 8,
            "channel_mult": [1, 2],
            "num_res_blocks": 1,
            "attention_resolutions": [7],
            "dropout": 0.0,
            "num_timesteps": 100,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "sample_dir": str(tmp_path / "samples"),
            "log_dir": str(tmp_path / "logs"),
            "seed": 0,
        }
    )

    base_train = get_mnist_dataset(config["data_dir"], train=True)
    base_val = get_mnist_dataset(config["data_dir"], train=False)
    gen = MaskGenerator(image_size=28, mask_types=[MaskType.CENTER])
    train_ds = InpaintingDataset(
        Subset(base_train, range(32)), gen, mask_type=MaskType.CENTER
    )
    val_ds = InpaintingDataset(
        Subset(base_val, range(16)), gen, mask_type=MaskType.CENTER
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = ConditionedUNet(
        image_channels=1,
        base_channels=config["base_channels"],
        channel_mult=tuple(config["channel_mult"]),
        num_res_blocks=config["num_res_blocks"],
        attention_resolutions=tuple(config["attention_resolutions"]),
        dropout=config["dropout"],
        image_size=28,
    )
    scheduler = NoiseScheduler(
        num_timesteps=config["num_timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = DDPMLoss()

    trainer = InpaintingTrainer(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    trainer.train()

    train_metrics_path = tmp_path / "logs" / "train_metrics.csv"
    val_metrics_path = tmp_path / "logs" / "val_metrics.csv"
    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"

    assert train_metrics_path.exists()
    assert val_metrics_path.exists()
    assert checkpoint_path.exists()
    assert (tmp_path / "checkpoints" / "epoch_001.pt").exists()

    with train_metrics_path.open("r", encoding="utf-8") as file:
        train_rows = list(csv.DictReader(file))

    assert len(train_rows) == 1
    assert torch.isfinite(torch.tensor(float(train_rows[0]["loss"])))

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "config" in checkpoint
