"""Tests for unconditional DDPM + RePaint path."""

from __future__ import annotations

from pathlib import Path

import torch
from generative_models.ddpm import NoiseScheduler
from generative_models.losses import DDPMLoss
from torch.utils.data import DataLoader, Subset

from image_inpainting.datasets import get_mnist_dataset
from image_inpainting.diffusion import inpaint, load_unconditional_checkpoint
from image_inpainting.models import UncondInpaintAdapter, build_unconditional_ddpm_from_config
from image_inpainting.utils import load_config


def _tiny_uncond_config(tmp_path: Path) -> dict:
    return {
        "training_mode": "unconditional",
        "image_channels": 1,
        "image_size": 28,
        "num_timesteps": 20,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "base_channels": 8,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [7],
        "dropout": 0.0,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "sample_dir": str(tmp_path / "samples"),
        "log_dir": str(tmp_path / "logs"),
        "epochs": 1,
        "learning_rate": 1e-3,
        "device": "cpu",
        "seed": 0,
    }


def test_uncond_adapter_ignores_conditioning_channels() -> None:
    config = {
        "image_channels": 1,
        "image_size": 28,
        "num_timesteps": 10,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "base_channels": 8,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [7],
        "dropout": 0.0,
    }
    ddpm = build_unconditional_ddpm_from_config(config).eval()
    adapter = UncondInpaintAdapter(ddpm).eval()

    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([3, 5])
    masked = torch.zeros_like(x)
    mask = torch.ones(2, 1, 28, 28)

    with torch.no_grad():
        a = adapter(x, t, masked, mask)
        b = ddpm.predict_noise(x, t)
    assert torch.allclose(a, b)


def test_uncond_inpaint_preserves_known_pixels() -> None:
    torch.manual_seed(0)
    config = {
        "image_channels": 1,
        "image_size": 28,
        "num_timesteps": 16,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "base_channels": 8,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [7],
        "dropout": 0.0,
    }
    ddpm = build_unconditional_ddpm_from_config(config).eval()
    adapter = UncondInpaintAdapter(ddpm).eval()
    scheduler = ddpm.scheduler

    original = torch.rand(2, 1, 28, 28)
    mask = torch.ones(2, 1, 28, 28)
    mask[:, :, 8:20, 8:20] = 0.0
    masked = original * mask

    result = inpaint(
        adapter,
        scheduler,
        masked,
        mask,
        original=original,
        jump_length=4,
        jump_n_sample=2,
        show_progress=False,
    )
    assert result.shape == original.shape
    assert ((result - original) * mask).abs().max().item() < 1e-5


def test_uncond_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = _tiny_uncond_config(tmp_path)
    model = build_unconditional_ddpm_from_config(config)
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(config["checkpoint_dir"]) / "epoch_001.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "metrics": {"loss": 0.1},
            "config": config,
        },
        ckpt_path,
    )

    adapter, scheduler, loaded_cfg, ckpt = load_unconditional_checkpoint(
        ckpt_path, device=torch.device("cpu")
    )
    assert ckpt["epoch"] == 1
    assert loaded_cfg["training_mode"] == "unconditional"
    assert isinstance(adapter, UncondInpaintAdapter)
    assert scheduler.num_timesteps == 20

    original = torch.rand(1, 1, 28, 28)
    mask = torch.ones(1, 1, 28, 28)
    mask[:, :, 10:18, 10:18] = 0.0
    result = inpaint(
        adapter,
        scheduler,
        original * mask,
        mask,
        original=original,
        jump_n_sample=1,
    )
    assert ((result - original) * mask).abs().max().item() < 1e-5


def test_uncond_one_train_step_mnist(tmp_path: Path) -> None:
    """Smoke: one optimizer step on a tiny MNIST subset (no masks)."""
    from generative_models.trainers import DDPMTrainer

    config = _tiny_uncond_config(tmp_path)
    config["batch_size"] = 8
    config["dataset"] = "MNIST"
    config["data_dir"] = "data/raw"

    base = get_mnist_dataset(config["data_dir"], train=True)
    loader = DataLoader(Subset(base, range(16)), batch_size=8, shuffle=True)
    model = build_unconditional_ddpm_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = DDPMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=DDPMLoss(),
        train_loader=loader,
        config=config,
        val_loader=None,
    )
    metrics = trainer.train_epoch()
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    trainer.save_checkpoint(1, metrics)
    assert (tmp_path / "checkpoints" / "latest.pt").exists()


def test_celeba_unconditional_config_loads() -> None:
    config = load_config("configs/celeba_unconditional.yaml")
    assert config["training_mode"] == "unconditional"
    assert int(config["image_channels"]) == 3
    assert int(config["image_size"]) == 64
    model = build_unconditional_ddpm_from_config(config)
    # Sanity: RGB U-Net in/out channels
    x = torch.randn(1, 3, 64, 64)
    t = torch.tensor([10])
    with torch.no_grad():
        out = model.predict_noise(x, t)
    assert out.shape == x.shape
