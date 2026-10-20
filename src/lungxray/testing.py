"""Helpers for tests and smoke runs. Bundles made here are untrained and meaningless."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from lungxray.data import IMAGENET_MEAN, IMAGENET_STD
from lungxray.model import build_meta, build_model, save_bundle


def make_random_bundle(
    path: str | Path,
    *,
    model_name: str = "efficientnet_b0",
    image_size: int = 224,
    resize_size: int = 256,
    seed: int = 0,
) -> Path:
    """Save a random-init binary bundle whose outputs still depend on the input."""
    torch.manual_seed(seed)
    model = build_model(model_name, num_classes=2, pretrained=False)
    _recalibrate_batchnorm(model, image_size)
    meta = build_meta(
        model_name=model_name,
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=image_size,
        resize_size=resize_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    meta["note"] = "random-init weights for testing; predictions are meaningless"
    return save_bundle(path, model, meta)


def _recalibrate_batchnorm(model: nn.Module, image_size: int, steps: int = 3) -> None:
    # With default init and BatchNorm running stats of mean 0 / var 1, activations shrink
    # ~30x per EfficientNet block in eval mode, so logits and Grad-CAM would be the same
    # for every image. Re-estimating the statistics on random inputs restores the scale.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None  # cumulative average over the calibration batches
    model.train()
    with torch.no_grad():
        for _ in range(steps):
            model(torch.randn(8, 3, image_size, image_size))
    model.eval()
