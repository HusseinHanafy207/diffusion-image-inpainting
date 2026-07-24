"""Inpainting trainer.

Training is almost identical to standard DDPM:

    original → mask → damaged image → forward diffuse → UNet → MSE(ε̂, ε)

Differences vs ``generative_models.trainers.DDPMTrainer``:
    - each batch already includes a fresh mask (from ``InpaintingDataset``)
    - U-Net sees ``(x_t, masked_image, mask)``
    - loss is still noise-prediction MSE (reuse ``DDPMLoss``)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from generative_models.ddpm import NoiseScheduler, forward_diffuse
from generative_models.utils.device import get_device


class InpaintingTrainer:
    """Orchestrates mask-conditioned DDPM training.

    Parameters
    ----------
    model:
        :class:`~image_inpainting.models.ConditionedUNet`.
    scheduler:
        ``generative_models.ddpm.NoiseScheduler``.
    optimizer:
        Torch optimizer.
    criterion:
        Noise-prediction loss (typically ``DDPMLoss``).
    train_loader:
        Yields ``(original, masked_image, mask)``.
    config:
        Experiment config (epochs, checkpoint paths, device, …).
    val_loader:
        Optional validation loader with the same batch layout.
    """

    TRAIN_FIELDS = ["epoch", "loss", "lr", "epoch_time"]
    VAL_FIELDS = ["epoch", "loss", "epoch_time"]

    def __init__(
        self,
        model: torch.nn.Module,
        scheduler: NoiseScheduler,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        config: dict[str, Any],
        val_loader: DataLoader | None = None,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = self._resolve_device(config.get("device", "auto"))
        self.model.to(self.device)
        # NoiseScheduler gathers coefficients onto the batch device in q_sample.

        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.sample_dir = Path(config["sample_dir"])
        self.log_dir = Path(config["log_dir"])

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_metrics_path = self.log_dir / config.get(
            "train_metrics_file", "train_metrics.csv"
        )
        self.val_metrics_path = self.log_dir / config.get(
            "val_metrics_file", "val_metrics.csv"
        )
        self.current_epoch = 0
        self._init_csv_logs()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return get_device()
        return torch.device(device)

    def _init_csv_logs(self) -> None:
        if not self.train_metrics_path.exists():
            with self.train_metrics_path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.TRAIN_FIELDS).writeheader()

        if self.val_loader is not None and not self.val_metrics_path.exists():
            with self.val_metrics_path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.VAL_FIELDS).writeheader()

    def _append_csv_row(
        self, path: Path, fieldnames: list[str], row: dict[str, Any]
    ) -> None:
        with path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fieldnames).writerow(row)

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(mode=training)

        total_loss = 0.0
        num_batches = 0
        first_batch_loss: float | None = None
        last_batch_loss: float | None = None

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for original, masked_image, mask in loader:
                original = original.to(self.device)
                masked_image = masked_image.to(self.device)
                mask = mask.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                # Forward diffuse the *clean* original; condition on damaged view + mask.
                x_t, t, noise = forward_diffuse(self.scheduler, original)
                noise_pred = self.model(x_t, t, masked_image, mask)
                loss = self.criterion(noise_pred, noise)

                if training:
                    loss.backward()
                    self.optimizer.step()

                batch_loss = float(loss.item())
                if first_batch_loss is None:
                    first_batch_loss = batch_loss
                last_batch_loss = batch_loss

                total_loss += batch_loss
                num_batches += 1

        metrics = {"loss": total_loss / max(num_batches, 1)}
        if training:
            metrics["first_batch_loss"] = first_batch_loss or 0.0
            metrics["last_batch_loss"] = last_batch_loss or 0.0
        return metrics

    def train_epoch(self) -> dict[str, float]:
        start_time = time.time()
        metrics = self._run_epoch(self.train_loader, training=True)
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["epoch_time"] = time.time() - start_time
        return metrics

    def validate(self) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        start_time = time.time()
        metrics = self._run_epoch(self.val_loader, training=False)
        metrics["epoch_time"] = time.time() - start_time
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        checkpoint_every = int(self.config.get("checkpoint_every", 5))
        total_epochs = int(self.config["epochs"])
        should_snapshot = (
            epoch == 1
            or epoch == total_epochs
            or (checkpoint_every > 0 and epoch % checkpoint_every == 0)
        )
        if should_snapshot:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = int(checkpoint["epoch"])
        return checkpoint

    def _print_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        epochs = self.config["epochs"]
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"Train Loss:  {train_metrics['loss']:.4f}")
        print(f"Time:        {train_metrics['epoch_time']:.1f} sec")
        if val_metrics is not None:
            print(f"Val Loss:    {val_metrics['loss']:.4f}")

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        self._append_csv_row(
            self.train_metrics_path,
            self.TRAIN_FIELDS,
            {
                "epoch": epoch,
                "loss": f"{train_metrics['loss']:.6f}",
                "lr": f"{train_metrics['lr']:.6f}",
                "epoch_time": f"{train_metrics['epoch_time']:.2f}",
            },
        )
        if val_metrics is not None:
            self._append_csv_row(
                self.val_metrics_path,
                self.VAL_FIELDS,
                {
                    "epoch": epoch,
                    "loss": f"{val_metrics['loss']:.6f}",
                    "epoch_time": f"{val_metrics['epoch_time']:.2f}",
                },
            )

    def train(self) -> None:
        if seed := self.config.get("seed"):
            torch.manual_seed(seed)

        start_epoch = int(self.config.get("start_epoch", 0))
        epochs = int(self.config["epochs"])

        for epoch in range(start_epoch + 1, epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self._print_metrics(epoch, train_metrics, val_metrics)
            self._log_metrics(epoch, train_metrics, val_metrics)
            self.save_checkpoint(epoch, train_metrics)
            print()
