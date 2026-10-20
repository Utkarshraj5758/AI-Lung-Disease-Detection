"""Load a model bundle once, then predict, explain (Grad-CAM) and sanity-check inputs."""

from __future__ import annotations

import io
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError

from lungxray.data import build_eval_transform, build_view_transform, to_rgb
from lungxray.dicom import dicom_to_pil
from lungxray.gradcam import grad_cam
from lungxray.model import load_bundle

DEFAULT_BORDERLINE_MARGIN = 0.10
_DICOM_SUFFIXES = {".dcm", ".dicom"}
# Basic input sanity check limits (0-255 scale). Warnings only; not an X-ray detector.
_MIN_GREY_STD = 8.0
_MAX_CHANNEL_DIFF = 15.0


@dataclass(frozen=True)
class Prediction:
    prob_pneumonia: float  # softmax of the raw logits
    prob_calibrated: float  # softmax of logits / temperature; the threshold applies to this
    label: str
    threshold: float
    borderline: bool  # |prob_calibrated - threshold| < borderline margin


def load_image(
    source: str | PathLike[str] | bytes | BinaryIO, filename: str | None = None
) -> Image.Image:
    """Open a PNG/JPEG/... or a DICOM file from a path, bytes or a binary file object.

    DICOM is detected from a ``.dcm``/``.dicom`` name or the ``DICM`` magic bytes.
    """
    if isinstance(source, (str, PathLike)):
        path = Path(source)
        filename = filename or path.name
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        data = source.read()

    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix in _DICOM_SUFFIXES or data[128:132] == b"DICM":
        return dicom_to_pil(data)
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as e:
        name = f" {filename!r}" if filename else ""
        raise ValueError(f"Unsupported or corrupt image{name}: {e}") from e
    return ImageOps.exif_transpose(image)


class Predictor:
    """Pneumonia classifier around one bundle. ``predict``/``explain`` are thread-safe.

    Preprocessing (crop size, resize, normalization), the decision threshold and the
    temperature all come from the bundle metadata.
    """

    def __init__(
        self,
        bundle_path: str | PathLike[str],
        device: str | torch.device | None = None,
        borderline_margin: float = DEFAULT_BORDERLINE_MARGIN,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.meta = load_bundle(bundle_path, self.device)
        self.class_names: list[str] = list(self.meta["class_names"])
        if len(self.class_names) != 2:
            raise ValueError(f"Predictor needs a binary model, got classes {self.class_names}")
        self.positive_index = self.class_names.index(self.meta["positive_class"])
        self.threshold = float(self.meta["threshold"])
        self.temperature = float(self.meta["temperature"])
        self.borderline_margin = borderline_margin
        size, resize = int(self.meta["image_size"]), int(self.meta["resize_size"])
        self._transform = build_eval_transform(size, resize, self.meta["mean"], self.meta["std"])
        self._view = build_view_transform(size, resize)

    @property
    def model_info(self) -> dict[str, Any]:
        keys = ("model_name", "git_commit", "created_at", "threshold", "temperature")
        return {key: self.meta[key] for key in keys}

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._transform(image).unsqueeze(0).to(self.device)

    def model_view(self, image: Image.Image) -> np.ndarray:
        """The RGB uint8 crop the model sees; Grad-CAM maps from :meth:`explain` align with it."""
        return np.asarray(self._view(image), dtype=np.uint8)

    def predict(self, image: Image.Image, threshold: float | None = None) -> Prediction:
        t = self._threshold(threshold)
        with torch.no_grad():
            logits = self.model(self.preprocess(image))[0].float()
        prob_raw = float(F.softmax(logits, dim=0)[self.positive_index])
        prob_cal = self._calibrated_prob(logits)
        return Prediction(
            prob_pneumonia=prob_raw,
            prob_calibrated=prob_cal,
            label=self.class_names[self._decide(prob_cal, t)],
            threshold=t,
            borderline=abs(prob_cal - t) < self.borderline_margin,
        )

    def explain(
        self,
        image: Image.Image,
        target: str | int = "predicted",
        threshold: float | None = None,
    ) -> np.ndarray:
        """Grad-CAM map (float32, ``image_size`` x ``image_size``, values in [0, 1]).

        ``target`` is ``"predicted"`` (the class :meth:`predict` would return with the
        same threshold), a class name, or a class index.
        """
        if target == "predicted":
            t = self._threshold(threshold)

            def target_class(logits: torch.Tensor) -> int:
                return self._decide(self._calibrated_prob(logits.float()), t)

            return grad_cam(self.model, self.preprocess(image), target_class)
        if isinstance(target, str):
            if target not in self.class_names:
                raise ValueError(f"Unknown target {target!r}; expected one of {self.class_names}")
            return grad_cam(self.model, self.preprocess(image), self.class_names.index(target))
        return grad_cam(self.model, self.preprocess(image), int(target))

    def check_input(self, image: Image.Image) -> list[str]:
        """Basic input sanity check: returns warnings, never blocks a prediction.

        Flags near-constant images, strong colour and tiny resolution. It does not
        detect whether the image is a chest X-ray.
        """
        warnings = []
        width, height = image.size
        size = int(self.meta["image_size"])
        if min(width, height) < size:
            warnings.append(
                f"Low resolution ({width}x{height}); the model input is {size}x{size}, "
                "so the image will be upscaled."
            )
        rgb = np.asarray(to_rgb(image), dtype=np.int16)
        grey_std = float(rgb.mean(axis=2).std())
        if grey_std < _MIN_GREY_STD:
            warnings.append(
                f"Image is nearly uniform (grey-level std {grey_std:.1f}); "
                "the prediction is unlikely to be meaningful."
            )
        channel_diff = max(
            float(np.abs(rgb[..., a] - rgb[..., b]).mean()) for a, b in ((0, 1), (0, 2), (1, 2))
        )
        if channel_diff > _MAX_CHANNEL_DIFF:
            warnings.append(
                f"Image has strong colour (mean channel difference {channel_diff:.1f}); "
                "chest X-rays are greyscale, so this may not be one."
            )
        return warnings

    def _threshold(self, threshold: float | None) -> float:
        t = self.threshold if threshold is None else float(threshold)
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {t}")
        return t

    def _calibrated_prob(self, logits: torch.Tensor) -> float:
        return float(F.softmax(logits / self.temperature, dim=0)[self.positive_index])

    def _decide(self, prob_calibrated: float, threshold: float) -> int:
        return self.positive_index if prob_calibrated >= threshold else 1 - self.positive_index
