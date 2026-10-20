"""Binary classification metrics, calibration error, threshold selection and bootstrap CIs.

Labels are 0/1 with 1 = positive (pneumonia); ``probs`` are P(positive). A sample is
predicted positive when ``prob >= threshold``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

METRIC_NAMES = (
    "accuracy",
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "ece",
)


def confusion_counts(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, int]:
    labels = np.asarray(labels).astype(bool)
    preds = np.asarray(probs) >= threshold
    return {
        "tn": int(np.sum(~labels & ~preds)),
        "fp": int(np.sum(~labels & preds)),
        "fn": int(np.sum(labels & ~preds)),
        "tp": int(np.sum(labels & preds)),
    }


def binary_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float, ece_bins: int = 15
) -> dict[str, float]:
    """Every name in :data:`METRIC_NAMES`; undefined values (e.g. AUROC with one class) are NaN."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    c = confusion_counts(labels, probs, threshold)
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    both_classes = labels.min() != labels.max() if labels.size else False
    return {
        "accuracy": _ratio(tp + tn, labels.size),
        "auroc": float(roc_auc_score(labels, probs)) if both_classes else float("nan"),
        "auprc": float(average_precision_score(labels, probs)) if both_classes else float("nan"),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "ece": expected_calibration_error(labels, probs, ece_bins),
    }


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """ECE of P(positive): bin-size-weighted mean |mean P - observed positive rate|.

    Uses ``n_bins`` equal-width bins on [0, 1].
    """
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0:
        return float("nan")
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(probs[in_bin].mean() - labels[in_bin].mean())
    return float(ece)


def reliability_curve(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per non-empty bin: mean predicted P(positive), observed positive rate, sample count."""
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    mean_pred, observed, counts = [], [], []
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            mean_pred.append(probs[in_bin].mean())
            observed.append(labels[in_bin].mean())
            counts.append(int(in_bin.sum()))
    return np.array(mean_pred), np.array(observed), np.array(counts)


def threshold_for_sensitivity(labels: np.ndarray, probs: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity (``prob >= t``) is at least ``target``."""
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target sensitivity must be in (0, 1], got {target}")
    positives = np.sort(np.asarray(probs, dtype=np.float64)[np.asarray(labels) == 1])[::-1]
    if positives.size == 0:
        raise ValueError("Cannot pick a sensitivity threshold without positive samples")
    # k positives must score >= t; the highest such t is the k-th largest positive score.
    k = min(max(int(np.ceil(target * positives.size - 1e-9)), 1), positives.size)
    return float(positives[k - 1])


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Threshold maximizing sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best = int(np.argmax(tpr - fpr))
    return float(min(thresholds[best], 1.0))  # roc_curve's first threshold is +inf


def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    ece_bins: int = 15,
    level: float = 0.95,
) -> dict[str, tuple[float, float] | None]:
    """Percentile bootstrap CIs for :data:`METRIC_NAMES` (resampling cases with replacement).

    Resamples where a metric is undefined (e.g. one class only) are skipped for that metric;
    a metric that is never defined gets ``None``.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_resamples):
        index = rng.integers(0, labels.size, labels.size)
        for name, value in binary_metrics(labels[index], probs[index], threshold, ece_bins).items():
            if np.isfinite(value):
                samples[name].append(value)
    tail = (1.0 - level) / 2 * 100
    return {
        name: (
            (
                float(np.percentile(samples[name], tail)),
                float(np.percentile(samples[name], 100 - tail)),
            )
            if samples[name]
            else None
        )
        for name in METRIC_NAMES
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")
