"""Temperature scaling and decision-threshold selection, both on the validation split only.

    python -m lungxray.calibrate --bundle artifacts/best.pt [--fast-dev-run] [--set k=v]

Fits T by minimizing validation NLL, then picks the highest threshold on the calibrated
P(pneumonia) that reaches ``eval.target_sensitivity``; both are written into the bundle.
The test split is never used here. ``train`` runs this automatically on its best bundle.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lungxray.config import Config, add_config_args, config_from_args
from lungxray.data import FAST_DEV_BATCH_SIZE, build_eval_transform, make_loader, split_dataset
from lungxray.metrics import (
    binary_metrics,
    expected_calibration_error,
    threshold_for_sensitivity,
    youden_threshold,
)
from lungxray.model import collect_logits, load_bundle, resolve_device, save_bundle

log = logging.getLogger(__name__)

TEMPERATURE_RANGE = (0.05, 20.0)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    """Temperature T > 0 minimizing the NLL of ``softmax(logits / T)`` (LBFGS on log T)."""
    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(*TEMPERATURE_RANGE))


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float64) / temperature, 1).numpy()


def calibrate_bundle(
    bundle_path: str | Path,
    config: Config,
    *,
    device: str | torch.device | None = None,
    fast_dev_run: bool = False,
) -> dict[str, Any]:
    """Fit T and the threshold on validation and write them into the bundle; returns a summary."""
    device = resolve_device(device)
    model, meta = load_bundle(bundle_path, device)
    transform = build_eval_transform(
        meta["image_size"], meta["resize_size"], meta["mean"], meta["std"]
    )
    dataset = split_dataset(
        config.paths.data_dir,
        "val",
        transform,
        fake=fast_dev_run,
        image_size=meta["image_size"],
        seed=config.train.seed,
    )
    loader = make_loader(
        dataset,
        FAST_DEV_BATCH_SIZE if fast_dev_run else config.data.batch_size,
        num_workers=0 if fast_dev_run else config.data.num_workers,
        device=device,
    )
    logits, labels = collect_logits(model, loader, device)
    positive = meta["class_names"].index(meta["positive_class"])
    y = (labels == positive).astype(int)

    temperature = fit_temperature(logits, labels)
    raw = softmax_np(logits)[:, positive]
    calibrated = softmax_np(logits, temperature)[:, positive]
    bins = config.eval.ece_bins
    threshold = threshold_for_sensitivity(y, calibrated, config.eval.target_sensitivity)
    at_threshold = binary_metrics(y, calibrated, threshold, bins)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    summary = {
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "temperature": temperature,
        "nll_before": float(F.cross_entropy(logits_t, labels_t)),
        "nll_after": float(F.cross_entropy(logits_t / temperature, labels_t)),
        "ece_before": expected_calibration_error(y, raw, bins),
        "ece_after": expected_calibration_error(y, calibrated, bins),
        "target_sensitivity": config.eval.target_sensitivity,
        "threshold": threshold,
        "sensitivity": at_threshold["sensitivity"],
        "specificity": at_threshold["specificity"],
        "auroc": at_threshold["auroc"],
        "youden_threshold": youden_threshold(y, calibrated),
    }
    metrics = {**meta.get("metrics", {}), "val": summary}
    save_bundle(
        bundle_path,
        model,
        {**meta, "temperature": temperature, "threshold": threshold, "metrics": metrics},
    )
    log.info(
        "Calibrated on val (n=%d): T=%.3f, ECE %.4f -> %.4f, threshold %.4f "
        "(sensitivity %.3f, specificity %.3f; Youden-J threshold %.4f)",
        summary["n"],
        temperature,
        summary["ece_before"],
        summary["ece_after"],
        threshold,
        summary["sensitivity"],
        summary["specificity"],
        summary["youden_threshold"],
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--bundle", type=Path, help="bundle to update (default: artifacts/best.pt)")
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="use synthetic FakeData instead of the dataset"
    )
    add_config_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bundle = args.bundle or Path(config.paths.artifacts_dir) / "best.pt"
    calibrate_bundle(bundle, config, device=args.device, fast_dev_run=args.fast_dev_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
