"""Image preprocessing shared by training, evaluation and inference.

Chest X-rays are greyscale; they are converted to 3-channel RGB (the grey value
repeated) so the ImageNet-pretrained weights can be reused unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_HIGH_BIT_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


def minmax_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale an array to the full uint8 range; constant input maps to 0."""
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo) * 255.0
    return np.round(np.nan_to_num(scaled)).astype(np.uint8)


def to_rgb(img: Image.Image) -> Image.Image:
    """Return an 8-bit RGB copy, min-max scaling 16-bit/float images first.

    PIL's own ``convert`` clips 16-bit values at 255 instead of scaling them.
    """
    if img.mode in _HIGH_BIT_DEPTH_MODES:
        img = Image.fromarray(minmax_to_uint8(np.asarray(img)))
    return img.convert("RGB")


class ToRGB:
    """Picklable transform for :func:`to_rgb` (DataLoader workers on Windows use spawn)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return to_rgb(img)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def build_view_transform(image_size: int, resize_size: int) -> transforms.Compose:
    """PIL -> PIL crop that the model sees, before tensor conversion and normalization.

    Grad-CAM maps line up with this crop, so overlays are drawn on it.
    """
    return transforms.Compose(
        [ToRGB(), transforms.Resize(resize_size), transforms.CenterCrop(image_size)]
    )


def build_eval_transform(
    image_size: int,
    resize_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Deterministic eval/inference transform: resize shorter side, center crop, normalize."""
    view = build_view_transform(image_size, resize_size)
    return transforms.Compose(
        [*view.transforms, transforms.ToTensor(), transforms.Normalize(list(mean), list(std))]
    )
