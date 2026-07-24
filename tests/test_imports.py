"""Scaffold / dependency smoke tests."""

from __future__ import annotations

import importlib

import pytest


def test_package_imports() -> None:
    import image_inpainting
    from image_inpainting.masks import MaskGenerator, MaskType
    from image_inpainting.datasets import (
        InpaintingDataset,
        apply_mask,
        get_mnist_inpainting_datasets,
    )
    from image_inpainting.utils import load_config

    assert image_inpainting.__version__
    assert MaskType.CENTER.value == "center"
    assert callable(MaskGenerator)
    assert callable(InpaintingDataset)
    assert callable(apply_mask)
    assert callable(get_mnist_inpainting_datasets)
    assert callable(load_config)


def test_generative_models_ddpm_available() -> None:
    """This project must import DDPM — not vendor a copy."""
    pytest.importorskip("generative_models")
    ddpm = importlib.import_module("generative_models.ddpm")
    assert hasattr(ddpm, "UNet")
    assert hasattr(ddpm, "NoiseScheduler")
    assert hasattr(ddpm, "forward_diffuse")
