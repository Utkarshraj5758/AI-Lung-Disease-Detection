"""Model construction and the checkpoint bundle (weights + plain-type metadata).

A bundle is a ``torch.save``'d dict ``{"format_version", "state_dict", "meta"}``. ``meta``
holds only plain Python types so bundles load with ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

BUNDLE_FORMAT_VERSION = 1
REQUIRED_META_KEYS = (
    "model_name",
    "class_names",
    "positive_class",
    "image_size",
    "resize_size",
    "mean",
    "std",
    "threshold",  # applies to the temperature-scaled P(positive_class)
    "temperature",
    "training_config",
    "git_commit",
    "dataset_fingerprint",
    "metrics",
    "created_at",
)
_PLAIN_SCALARS = (str, int, float, bool, type(None))


def build_model(
    name: str = "efficientnet_b0", num_classes: int = 2, pretrained: bool = True
) -> nn.Module:
    """A torchvision EfficientNet with its last linear layer resized to ``num_classes``."""
    if not name.startswith("efficientnet_"):
        raise ValueError(f"Unsupported model {name!r}: only torchvision EfficientNets work")
    try:
        model = models.get_model(name, weights="DEFAULT" if pretrained else None)
    except ValueError as e:
        raise ValueError(f"Unknown torchvision model {name!r}") from e
    head = model.classifier[-1]
    model.classifier[-1] = nn.Linear(head.in_features, num_classes)
    return model


def build_meta(
    *,
    model_name: str,
    class_names: Sequence[str],
    image_size: int,
    resize_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    positive_class: str = "PNEUMONIA",
    threshold: float = 0.5,
    temperature: float = 1.0,
    training_config: Mapping[str, Any] | None = None,
    dataset_fingerprint: str | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bundle metadata with the git commit and a UTC creation timestamp filled in."""
    if positive_class not in class_names:
        raise ValueError(f"positive_class {positive_class!r} is not in {list(class_names)}")
    return {
        "model_name": model_name,
        "class_names": list(class_names),
        "positive_class": positive_class,
        "image_size": int(image_size),
        "resize_size": int(resize_size),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "threshold": float(threshold),
        "temperature": float(temperature),
        "training_config": dict(training_config) if training_config is not None else None,
        "git_commit": git_commit(),
        "dataset_fingerprint": dataset_fingerprint,
        "metrics": dict(metrics or {}),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def save_bundle(path: str | Path, model: nn.Module, meta: Mapping[str, Any]) -> Path:
    """Write weights and metadata atomically; returns the path."""
    meta = dict(meta)
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"Bundle meta is missing required keys: {missing}")
    _check_plain(meta, "meta")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {"format_version": BUNDLE_FORMAT_VERSION, "state_dict": state_dict, "meta": meta}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_bundle(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a bundle into an eval-mode model on ``device``; fails loudly on bad input."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found: {path}. Pass the path to a bundle written by "
            "lungxray (e.g. artifacts/best.pt)."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(f"Could not load model bundle {path}: {e}") from e
    if not isinstance(payload, dict) or not {"state_dict", "meta"} <= payload.keys():
        raise ValueError(
            f"{path} is not a lungxray bundle (expected 'state_dict' and 'meta' keys). "
            "Plain state_dict files such as the legacy lung_model.pth must be re-saved "
            "with lungxray.model.save_bundle."
        )
    version = payload.get("format_version")
    if version != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"{path} has bundle format version {version!r}; expected {BUNDLE_FORMAT_VERSION}"
        )
    meta = payload["meta"]
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"{path} meta is missing required keys: {missing}")
    model = build_model(meta["model_name"], num_classes=len(meta["class_names"]), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, meta


def git_commit() -> str | None:
    """HEAD commit of the source checkout (``-dirty`` if tracked files changed), else None."""
    cwd = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = head.stdout.strip()
    if head.returncode != 0 or not commit:
        return None
    return commit + ("-dirty" if status.stdout.strip() else "")


def _check_plain(obj: Any, where: str) -> None:
    # Exact type checks: numpy scalars subclass float but are rejected by weights_only.
    if type(obj) in _PLAIN_SCALARS:
        return
    if type(obj) in (list, tuple):
        for i, value in enumerate(obj):
            _check_plain(value, f"{where}[{i}]")
        return
    if type(obj) is dict:
        for key, value in obj.items():
            if type(key) is not str:
                raise TypeError(f"{where} has a non-string key {key!r}")
            _check_plain(value, f"{where}.{key}")
        return
    raise TypeError(
        f"{where} has type {type(obj).__name__}; bundle metadata must be plain "
        "str/int/float/bool/None/list/dict so it loads with weights_only=True"
    )
