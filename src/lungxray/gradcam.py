"""Hook-free, thread-safe Grad-CAM for torchvision EfficientNet-style models.

The backbone (``model.features``) runs once under ``torch.no_grad()``. Only the small
head is re-run with autograd, and the gradient is taken with ``torch.autograd.grad``
w.r.t. the detached feature map. No hooks are registered and no parameter ``.grad`` is
written, so one eval-mode model can serve concurrent requests.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def grad_cam(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int | Callable[[torch.Tensor], int] | None = None,
) -> np.ndarray:
    """Grad-CAM for a single image.

    Args:
        model: model with ``features``, ``avgpool`` and ``classifier``, in eval mode.
        x: normalized input of shape ``(1, 3, H, W)`` on the model's device.
        target_class: class index to explain; or a callable that maps this image's
            detached 1-D logits to an index; or ``None`` for the arg-max class.

    Returns:
        float32 array of shape ``(H, W)`` with values in ``[0, 1]``.
    """
    _require_supported(model)
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected input of shape (1, C, H, W), got {tuple(x.shape)}")

    with torch.no_grad():
        feats = model.features(x)
    # enable_grad: callers may be inside a no_grad block. (inference_mode would not work.)
    with torch.enable_grad():
        feats = feats.detach().requires_grad_(True)
        logits = model.classifier(torch.flatten(model.avgpool(feats), 1))
        target = _resolve_target(target_class, logits[0].detach())
        (grads,) = torch.autograd.grad(logits[0, target], feats)

    with torch.no_grad():
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
        cam = cam / cam.amax().clamp_min(1e-8)  # an all-zero map stays zero, no NaN
        cam = F.interpolate(cam, size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False)
    return cam[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a JET-coloured CAM onto an RGB uint8 image (high CAM = red); returns RGB uint8."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 uint8 RGB image, got {rgb.dtype} {rgb.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    height, width = rgb.shape[:2]
    cam = np.clip(np.nan_to_num(np.asarray(cam, dtype=np.float32)), 0.0, 1.0)
    if cam.shape != (height, width):
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.round(cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)  # OpenCV colour maps are BGR
    return cv2.addWeighted(rgb, 1.0 - alpha, heat, alpha, 0.0)


def _require_supported(model: nn.Module) -> None:
    if not all(hasattr(model, name) for name in ("features", "avgpool", "classifier")):
        raise TypeError(
            "grad_cam needs a model with .features, .avgpool and .classifier "
            f"(torchvision EfficientNet); got {type(model).__name__}"
        )
    if model.training:
        raise ValueError("Call model.eval() before grad_cam (switching modes is not thread-safe)")


def _resolve_target(
    target_class: int | Callable[[torch.Tensor], int] | None, logits: torch.Tensor
) -> int:
    if target_class is None:
        index = int(logits.argmax())
    elif callable(target_class):
        index = int(target_class(logits))
    else:
        index = int(target_class)
    if not 0 <= index < logits.shape[0]:
        raise ValueError(f"Target class {index} out of range for {logits.shape[0]} classes")
    return index
