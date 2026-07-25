"""Unit tests for damaged-image creation and InpaintingDataset."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from image_inpainting.datasets import (
    InpaintingDataset,
    apply_mask,
    get_fashion_mnist_inpainting_dataloaders,
    get_mnist_inpainting_dataloaders,
)
from image_inpainting.masks import MaskGenerator, MaskType


def test_apply_mask_zeros_missing_keeps_known() -> None:
    image = torch.ones(1, 4, 4)
    mask = torch.ones(1, 4, 4)
    mask[:, 1:3, 1:3] = 0.0
    damaged = apply_mask(image, mask)
    assert damaged[0, 0, 0].item() == 1.0
    assert damaged[0, 1, 1].item() == 0.0
    assert torch.equal(damaged, image * mask)


def test_apply_mask_batch_broadcast() -> None:
    images = torch.rand(2, 1, 8, 8)
    masks = torch.ones(2, 1, 8, 8)
    masks[:, :, 2:6, 2:6] = 0.0
    damaged = apply_mask(images, masks)
    assert damaged.shape == images.shape
    assert (damaged[:, :, 2:6, 2:6] == 0).all()


def test_inpainting_dataset_center_missing_middle() -> None:
    """Digit with a center mask → missing middle (Phase 2 example)."""
    torch.manual_seed(0)
    # Synthetic "digit": bright block covering the whole canvas.
    images = torch.ones(5, 1, 28, 28)
    labels = torch.zeros(5, dtype=torch.long)
    base = TensorDataset(images, labels)
    gen = MaskGenerator(image_size=28, center_ratio=0.4)
    ds = InpaintingDataset(base, gen, mask_type=MaskType.CENTER)

    original, masked, mask = ds[0]
    assert original.shape == (1, 28, 28)
    assert masked.shape == (1, 28, 28)
    assert mask.shape == (1, 28, 28)

    assert mask[0, 0, 0].item() == 1.0
    assert mask[0, 14, 14].item() == 0.0
    assert masked[0, 0, 0].item() == 1.0
    assert masked[0, 14, 14].item() == 0.0
    assert torch.equal(masked, apply_mask(original, mask))


def test_inpainting_dataset_return_label() -> None:
    images = torch.rand(3, 1, 28, 28)
    labels = torch.tensor([8, 3, 1])
    base = TensorDataset(images, labels)
    ds = InpaintingDataset(
        base,
        MaskGenerator(image_size=28),
        mask_type="rectangle",
        return_label=True,
    )
    original, masked, mask, label = ds[0]
    assert label.item() == 8
    assert original.shape[0] == 1
    assert masked.shape == original.shape
    assert mask.shape == (1, 28, 28)


def test_mnist_inpainting_dataloader_shapes() -> None:
    torch.manual_seed(0)
    train_loader, _ = get_mnist_inpainting_dataloaders(
        batch_size=8,
        data_dir="data/raw",
        mask_generator=MaskGenerator(image_size=28, mask_types=[MaskType.CENTER]),
        mask_type=MaskType.CENTER,
    )
    original, masked, mask = next(iter(train_loader))
    assert original.shape == (8, 1, 28, 28)
    assert masked.shape == (8, 1, 28, 28)
    assert mask.shape == (8, 1, 28, 28)
    assert (masked == original * mask).all()


def test_fashion_mnist_inpainting_dataloader_shapes() -> None:
    torch.manual_seed(0)
    train_loader, _ = get_fashion_mnist_inpainting_dataloaders(
        batch_size=8,
        data_dir="data/raw",
        mask_generator=MaskGenerator(image_size=28, mask_types=[MaskType.CENTER]),
        mask_type=MaskType.CENTER,
    )
    original, masked, mask = next(iter(train_loader))
    assert original.shape == (8, 1, 28, 28)
    assert masked.shape == (8, 1, 28, 28)
    assert mask.shape == (8, 1, 28, 28)
    assert (masked == original * mask).all()


def test_factory_dispatches_fashion_mnist() -> None:
    from image_inpainting.datasets import get_base_dataset, normalize_dataset_name

    assert normalize_dataset_name("Fashion-MNIST") == "fashionmnist"
    ds = get_base_dataset("Fashion-MNIST", "data/raw", train=False)
    image, _label = ds[0]
    assert tuple(image.shape) == (1, 28, 28)


def test_dataloader_collate_three_tensors() -> None:
    images = torch.rand(4, 1, 16, 16)
    base = TensorDataset(images, torch.zeros(4, dtype=torch.long))
    ds = InpaintingDataset(base, MaskGenerator(image_size=16), mask_type="holes")
    loader = DataLoader(ds, batch_size=4)
    original, masked, mask = next(iter(loader))
    assert original.shape[0] == 4
    assert masked.shape[0] == 4
    assert mask.shape[0] == 4
