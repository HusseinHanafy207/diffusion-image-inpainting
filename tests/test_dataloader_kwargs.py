"""Tests for DataLoader worker / pin_memory helpers."""

from __future__ import annotations

from image_inpainting.datasets.loader_utils import build_dataloader_kwargs
from image_inpainting.datasets import get_mnist_inpainting_dataloaders
from image_inpainting.masks import MaskGenerator, MaskType
from image_inpainting.utils import load_config


def test_build_dataloader_kwargs_zero_workers() -> None:
    kwargs = build_dataloader_kwargs(num_workers=0, pin_memory=True)
    assert kwargs == {"num_workers": 0, "pin_memory": True}
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_build_dataloader_kwargs_with_workers() -> None:
    kwargs = build_dataloader_kwargs(num_workers=2, pin_memory=True)
    assert kwargs["num_workers"] == 2
    assert kwargs["pin_memory"] is True
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 2


def test_build_dataloader_kwargs_can_disable_persistent_workers() -> None:
    kwargs = build_dataloader_kwargs(
        num_workers=2, pin_memory=True, persistent_workers=False
    )
    assert kwargs["persistent_workers"] is False


def test_curriculum_config_disables_persistent_workers() -> None:
    from image_inpainting.datasets import get_inpainting_dataloaders_from_config
    from image_inpainting.masks import build_mask_generator_from_config

    phase4 = load_config("configs/celeba_phase4_curriculum.yaml")
    assert phase4.get("center_ratio_schedule")
    assert phase4.get("persistent_workers") is False

    # MNIST avoids CelebA download while checking from_config wiring.
    config = load_config("configs/mnist.yaml")
    config["center_ratio_schedule"] = phase4["center_ratio_schedule"]
    config["num_workers"] = 2
    config["pin_memory"] = False
    config["batch_size"] = 4
    train_loader, _ = get_inpainting_dataloaders_from_config(
        config,
        build_mask_generator_from_config(config),
        num_workers=2,
        pin_memory=False,
    )
    assert train_loader.num_workers == 2
    assert train_loader.persistent_workers is False


def test_celeba_config_has_workers() -> None:
    config = load_config("configs/celeba.yaml")
    assert int(config["num_workers"]) == 2
    assert bool(config["pin_memory"]) is True


def test_mnist_loader_accepts_pin_memory() -> None:
    train_loader, _ = get_mnist_inpainting_dataloaders(
        batch_size=4,
        data_dir="data/raw",
        mask_generator=MaskGenerator(image_size=28, mask_types=[MaskType.CENTER]),
        mask_type=MaskType.CENTER,
        num_workers=0,
        pin_memory=False,
    )
    assert train_loader.num_workers == 0
    assert train_loader.pin_memory is False
    original, masked, mask = next(iter(train_loader))
    assert original.shape[0] == 4
    assert masked.shape == original.shape
    assert mask.shape[1] == 1
