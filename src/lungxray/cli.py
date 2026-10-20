"""Command line: ``lungxray predict IMAGE [--cam OUT.png] [--json]``. Research use only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from lungxray.gradcam import overlay
from lungxray.inference import Predictor, load_image

DEFAULT_WEIGHTS = Path("artifacts/best.pt")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _predict(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lungxray",
        description="Pneumonia vs normal chest X-ray classifier. Research use only; "
        "not a medical device.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="classify one image (PNG, JPEG or DICOM)")
    predict.add_argument("image", type=Path)
    predict.add_argument(
        "--weights",
        type=Path,
        default=Path(os.environ.get("LUNGXRAY_WEIGHTS", DEFAULT_WEIGHTS)),
        help=f"model bundle (default: $LUNGXRAY_WEIGHTS or {DEFAULT_WEIGHTS})",
    )
    predict.add_argument("--cam", type=Path, metavar="OUT.png", help="write a Grad-CAM overlay")
    predict.add_argument("--alpha", type=float, default=0.4, help="overlay opacity (default 0.4)")
    predict.add_argument("--threshold", type=float, help="override the bundle's threshold")
    predict.add_argument("--device", help="torch device, e.g. cpu or cuda (default: auto)")
    predict.add_argument("--json", action="store_true", help="print JSON instead of text")
    return parser


def _predict(args: argparse.Namespace) -> int:
    image = load_image(args.image)
    predictor = Predictor(args.weights, device=args.device)
    prediction = predictor.predict(image, threshold=args.threshold)
    warnings = predictor.check_input(image)

    cam_path = None
    if args.cam:
        cam = predictor.explain(image, threshold=args.threshold)
        args.cam.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay(predictor.model_view(image), cam, args.alpha)).save(args.cam)
        cam_path = str(args.cam)

    if args.json:
        result = {
            "file": str(args.image),
            "prediction": asdict(prediction),
            "input_warnings": warnings,
            "cam": cam_path,
            "model": predictor.model_info,
        }
        print(json.dumps(result, indent=2))
        return 0

    if predictor.temperature == 1.0:
        detail = "uncalibrated: no temperature fitted"
    else:
        detail = f"temperature {predictor.temperature:.3f}; raw {prediction.prob_pneumonia:.4f}"
    flag = "  [borderline]" if prediction.borderline else ""
    print(
        f"{args.image}: {prediction.label}  P(pneumonia)={prediction.prob_calibrated:.4f} "
        f"({detail}), threshold {prediction.threshold:.2f}{flag}"
    )
    for warning in warnings:
        print(f"warning: {warning}")
    if cam_path:
        print(f"Grad-CAM overlay written to {cam_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
