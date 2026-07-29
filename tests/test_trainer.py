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
from image_inpainting.trainers import InpaintingTrainer, resolve_center_ratio_for_epoch
from image_inpainting.utils import load_config


def _tiny_trainer_setup(
    tmp_path: Path,
    *,
    hole_loss_weight: float = 0.0,
    center_ratio_schedule: list[dict] | None = None,
    epochs: int = 1,
    center_ratio: float = 0.4,
) -> tuple[InpaintingTrainer, MaskGenerator]:
    config = load_config("configs/mnist.yaml")
    config.update(
        {
            "epochs": epochs,
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
            "hole_loss_weight": hole_loss_weight,
            "center_ratio": center_ratio,
        }
    )
    if center_ratio_schedule is not None:
        config["center_ratio_schedule"] = center_ratio_schedule

    base_train = get_mnist_dataset(config["data_dir"], train=True)
    base_val = get_mnist_dataset(config["data_dir"], train=False)
    gen = MaskGenerator(
        image_size=28,
        mask_types=[MaskType.CENTER],
        center_ratio=center_ratio,
    )
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
        mask_generator=gen,
    )
    return trainer, gen


def test_inpainting_trainer_runs_one_epoch(tmp_path: Path) -> None:
    trainer, _ = _tiny_trainer_setup(tmp_path)
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


def test_hole_weighted_loss_runs_one_epoch(tmp_path: Path) -> None:
    trainer, _ = _tiny_trainer_setup(tmp_path, hole_loss_weight=4.0)
    assert trainer.hole_loss_weight == 4.0
    trainer.train()
    checkpoint = torch.load(tmp_path / "checkpoints" / "latest.pt", weights_only=False)
    assert torch.isfinite(torch.tensor(float(checkpoint["metrics"]["loss"])))


def test_compute_loss_weights_hole_more_than_known(tmp_path: Path) -> None:
    """Hole pixels (m=0) should contribute more when λ > 0."""
    noise = torch.zeros(1, 1, 4, 4)
    noise_pred = torch.ones(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    mask[:, :, 1:3, 1:3] = 0.0  # 4 hole pixels out of 16

    trainer, _ = _tiny_trainer_setup(tmp_path, hole_loss_weight=4.0)
    loss = trainer._compute_loss(noise_pred, noise, mask)
    # All residuals are 1; mean weight = (12*1 + 4*5) / 16 = 2.0
    assert abs(loss.item() - 2.0) < 1e-5


def test_resolve_center_ratio_for_epoch() -> None:
    schedule = [
        {"until_epoch": 37, "center_ratio": 0.25},
        {"until_epoch": 44, "center_ratio": 0.32},
        {"until_epoch": 50, "center_ratio": 0.40},
    ]
    assert resolve_center_ratio_for_epoch(schedule, 31) == 0.25
    assert resolve_center_ratio_for_epoch(schedule, 37) == 0.25
    assert resolve_center_ratio_for_epoch(schedule, 38) == 0.32
    assert resolve_center_ratio_for_epoch(schedule, 44) == 0.32
    assert resolve_center_ratio_for_epoch(schedule, 45) == 0.40
    assert resolve_center_ratio_for_epoch(schedule, 99) == 0.40
    assert resolve_center_ratio_for_epoch(None, 10, default=0.4) == 0.4


def test_center_ratio_curriculum_updates_generator(tmp_path: Path) -> None:
    schedule = [
        {"until_epoch": 1, "center_ratio": 0.25},
        {"until_epoch": 2, "center_ratio": 0.40},
    ]
    trainer, gen = _tiny_trainer_setup(
        tmp_path,
        center_ratio_schedule=schedule,
        epochs=2,
        center_ratio=0.25,
    )
    assert gen.center_ratio == 0.25
    trainer.train()
    assert abs(gen.center_ratio - 0.40) < 1e-9
