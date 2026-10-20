"""Evaluate a bundle on the test split with its frozen threshold and temperature.

    python -m lungxray.evaluate [--bundle artifacts/best.pt] [--fast-dev-run] [--set k=v]

Writes ``<reports_dir>/metrics.json`` (point estimates with 95% bootstrap CIs) and
confusion_matrix / roc / pr / reliability / gradcam_examples PNGs, and stores the test
summary in the bundle's ``meta["metrics"]["test"]``. Nothing is tuned on the test split.

Because the dataset folders are used as-is, the report also gives the patient overlap
between splits and the same metrics on test images whose patient is not in train.
``--fast-dev-run`` evaluates a random-init bundle on synthetic FakeData and writes next
to that bundle, never to ``reports/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

from lungxray.calibrate import softmax_np
from lungxray.config import Config, add_config_args, config_from_args
from lungxray.data import (
    FAST_DEV_BATCH_SIZE,
    build_eval_transform,
    dataset_fingerprint,
    make_loader,
    patient_overlap,
    raw_image,
    sample_patients,
    scan_dataset,
    split_dataset,
)
from lungxray.gradcam import overlay
from lungxray.inference import Predictor
from lungxray.metrics import (
    METRIC_NAMES,
    binary_metrics,
    bootstrap_ci,
    confusion_counts,
    expected_calibration_error,
    reliability_curve,
)
from lungxray.model import collect_logits, git_commit, resolve_device, save_bundle
from lungxray.testing import make_random_bundle

log = logging.getLogger(__name__)

# Reference data-viz palette (light surface): text tokens, recessive neutrals, two series.
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DPI = 150


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a bundle on the test split.")
    parser.add_argument("--bundle", type=Path, help="default: <artifacts_dir>/best.pt")
    parser.add_argument("--reports-dir", type=Path, help="default: <reports_dir> from config")
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="random-init bundle (unless --bundle) on synthetic FakeData",
    )
    add_config_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = evaluate(
        config,
        bundle_path=args.bundle,
        device=args.device,
        fast_dev_run=args.fast_dev_run,
        reports_dir=args.reports_dir,
    )
    print(f"Reports written to {report['reports_dir']}")
    return 0


def evaluate(
    config: Config,
    *,
    bundle_path: str | Path | None = None,
    device: str | None = None,
    fast_dev_run: bool = False,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(device)
    created_bundle = False
    if bundle_path is None and fast_dev_run:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        bundle_path = make_random_bundle(
            Path(config.paths.runs_dir) / f"{stamp}-fast-dev-run-eval" / "random.pt",
            model_name=config.model.name,
            image_size=config.data.image_size,
            resize_size=config.data.resize_size,
            seed=config.train.seed,
        )
        created_bundle = True
    bundle_path = Path(bundle_path or Path(config.paths.artifacts_dir) / "best.pt")
    if reports_dir is None:
        reports_dir = bundle_path.parent / "reports" if fast_dev_run else config.paths.reports_dir
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    predictor = Predictor(bundle_path, device=device)
    meta = predictor.meta
    transform = build_eval_transform(
        meta["image_size"], meta["resize_size"], meta["mean"], meta["std"]
    )
    dataset = split_dataset(
        config.paths.data_dir,
        "test",
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
    logits, labels = collect_logits(predictor.model, loader, device)
    positive = predictor.positive_index
    y = (labels == positive).astype(int)
    raw = softmax_np(logits)[:, positive]
    calibrated = softmax_np(logits, predictor.temperature)[:, positive]
    threshold = predictor.threshold
    bins = config.eval.ece_bins
    ci_args = {"n_resamples": config.eval.bootstrap_resamples, "seed": config.train.seed}

    report: dict[str, Any] = {
        "split": "test",
        "fast_dev_run": fast_dev_run,
        "n": _class_counts(y, predictor.class_names, positive),
        "threshold": threshold,
        "temperature": predictor.temperature,
        "decision_rule": "PNEUMONIA if temperature-scaled P(pneumonia) >= threshold; "
        "threshold and temperature were fitted on the validation split",
        "metrics": _metrics_with_ci(y, calibrated, threshold, bins, ci_args),
        "confusion_matrix": confusion_counts(y, calibrated, threshold),
        "ece_uncalibrated": expected_calibration_error(y, raw, bins),
        "ece_definition": f"{bins} equal-width bins on P(pneumonia)",
        "bootstrap": {
            "resamples": ci_args["n_resamples"],
            "seed": ci_args["seed"],
            "level": 0.95,
            "method": "percentile; test cases resampled with replacement",
        },
        "validation": meta["metrics"].get("val"),
        "model": {
            "bundle": str(bundle_path),
            "model_name": meta["model_name"],
            "git_commit": meta["git_commit"],
            "created_at": meta["created_at"],
            "dataset_fingerprint": meta["dataset_fingerprint"],
        },
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "evaluation_git_commit": git_commit(),
    }
    if not fast_dev_run:
        report.update(
            _leakage_report(config, dataset, predictor, y, calibrated, threshold, bins, ci_args)
        )
        if report["dataset_fingerprint"] != meta["dataset_fingerprint"]:
            log.warning(
                "Dataset fingerprint %s differs from the one the bundle was trained on (%s)",
                report["dataset_fingerprint"],
                meta["dataset_fingerprint"],
            )

    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    point = {name: report["metrics"][name]["value"] for name in METRIC_NAMES}
    plot_confusion_matrix(report["confusion_matrix"], predictor.class_names, threshold, reports_dir)
    plot_roc(y, calibrated, point, reports_dir)
    plot_pr(y, calibrated, point, reports_dir)
    plot_reliability(y, raw, calibrated, bins, predictor.temperature, reports_dir)
    plot_gradcam_examples(predictor, dataset, y, calibrated, threshold, reports_dir)

    if not fast_dev_run or created_bundle:
        summary = {
            "n": report["n"],
            "threshold": threshold,
            "temperature": predictor.temperature,
            **{name: _finite_or_none(value) for name, value in point.items()},
            "evaluated_at": report["evaluated_at"],
        }
        if "test_unseen_patients" in report:
            subset = report["test_unseen_patients"]["metrics"]
            summary["unseen_patients"] = {
                name: _finite_or_none(subset[name]["value"]) for name in METRIC_NAMES
            }
        new_meta = {**meta, "metrics": {**meta["metrics"], "test": summary}}
        save_bundle(bundle_path, predictor.model, new_meta)

    _log_summary(report)
    report["reports_dir"] = str(reports_dir)
    return report


def _leakage_report(
    config: Config,
    dataset,
    predictor: Predictor,
    y: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    bins: int,
    ci_args: dict[str, int],
) -> dict[str, Any]:
    records = scan_dataset(config.paths.data_dir)
    train_patients = {r.patient for r in records if r.split == "train" and r.patient is not None}
    patients = sample_patients(dataset)
    unseen = np.array([p is not None and p not in train_patients for p in patients])
    return {
        "dataset_fingerprint": dataset_fingerprint(records),
        "patient_overlap": patient_overlap(records),
        "test_unseen_patients": {
            "description": "test images whose patient id does not appear in the train split",
            "n": _class_counts(y[unseen], predictor.class_names, predictor.positive_index),
            "metrics": _metrics_with_ci(y[unseen], probs[unseen], threshold, bins, ci_args),
            "confusion_matrix": confusion_counts(y[unseen], probs[unseen], threshold),
        },
    }


def _metrics_with_ci(
    y: np.ndarray, probs: np.ndarray, threshold: float, bins: int, ci_args: dict[str, int]
) -> dict[str, dict[str, Any]]:
    point = binary_metrics(y, probs, threshold, bins)
    ci = bootstrap_ci(y, probs, threshold, ece_bins=bins, **ci_args)
    return {name: {"value": point[name], "ci95": ci[name]} for name in METRIC_NAMES}


def _class_counts(y: np.ndarray, class_names: Sequence[str], positive: int) -> dict[str, int]:
    negative = 1 - positive
    return {
        "total": int(y.size),
        class_names[negative]: int((y == 0).sum()),
        class_names[positive]: int((y == 1).sum()),
    }


def _log_summary(report: dict[str, Any]) -> None:
    def fmt(entry: dict[str, Any]) -> str:
        value, ci = entry["value"], entry["ci95"]
        if value is None or not math.isfinite(value):
            return "n/a"
        return f"{value:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")

    log.info(
        "Test n=%s, threshold %.4f, T %.3f",
        report["n"],
        report["threshold"],
        report["temperature"],
    )
    for name in METRIC_NAMES:
        line = f"  {name:12s} {fmt(report['metrics'][name])}"
        if "test_unseen_patients" in report:
            line += (
                f"   unseen-patient subset: {fmt(report['test_unseen_patients']['metrics'][name])}"
            )
        log.info(line)


# ---------------------------------------------------------------------------
# Plots (matplotlib Figure objects directly: no pyplot state, no GUI backend)
# ---------------------------------------------------------------------------


def _figure(width: float = 5.2, height: float = 4.6) -> Figure:
    return Figure(figsize=(width, height), dpi=DPI, facecolor=SURFACE)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", color=TEXT, fontsize=10, fontweight="bold", pad=18)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=TEXT_2, fontsize=8)


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")


def plot_confusion_matrix(
    counts: dict[str, int], class_names: Sequence[str], threshold: float, out_dir: Path
) -> None:
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    rows = matrix.sum(axis=1, keepdims=True)
    rates = np.divide(matrix, rows, out=np.zeros_like(matrix, dtype=float), where=rows > 0)
    fig = _figure(4.6, 4.2)
    ax = fig.add_subplot()
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
    names = list(class_names)
    for i in range(2):
        for j in range(2):
            ink = SURFACE if rates[i, j] > 0.55 else TEXT
            ax.text(
                j, i - 0.08, f"{matrix[i, j]}", ha="center", va="center", color=ink, fontsize=14
            )
            ax.text(
                j,
                i + 0.18,
                f"{rates[i, j]:.1%} of row",
                ha="center",
                va="center",
                color=ink,
                fontsize=8,
            )
    ax.set_xticks([0, 1], [f"Predicted {n}" for n in names])
    ax.set_yticks([0, 1], [f"Actual {n}" for n in names], rotation=90, va="center")
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _title(ax, "Confusion matrix (test)", f"Threshold {threshold:.3f} on calibrated P(pneumonia)")
    _save(fig, out_dir / "confusion_matrix.png")


def plot_roc(y: np.ndarray, probs: np.ndarray, point: dict[str, float], out_dir: Path) -> None:
    fig = _figure()
    ax = fig.add_subplot()
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.62, 0.55, "chance", color=MUTED, fontsize=8, rotation=37)
    if len(set(y.tolist())) == 2:
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(fpr, tpr, color=SERIES_1, linewidth=2)
    fpr_op, tpr_op = 1 - point["specificity"], point["sensitivity"]
    if math.isfinite(fpr_op) and math.isfinite(tpr_op):
        _operating_point(ax, fpr_op, tpr_op, "operating point")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    _title(ax, "ROC curve (test)", f"AUROC {_fmt(point['auroc'])}")
    _save(fig, out_dir / "roc.png")


def plot_pr(y: np.ndarray, probs: np.ndarray, point: dict[str, float], out_dir: Path) -> None:
    fig = _figure()
    ax = fig.add_subplot()
    _style(ax)
    prevalence = float(np.mean(y)) if y.size else 0.0
    ax.axhline(prevalence, color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.02, prevalence + 0.015, f"prevalence {prevalence:.2f}", color=MUTED, fontsize=8)
    if len(set(y.tolist())) == 2:
        precision, recall, _ = precision_recall_curve(y, probs)
        ax.plot(recall, precision, color=SERIES_1, linewidth=2)
    if math.isfinite(point["sensitivity"]) and math.isfinite(point["ppv"]):
        _operating_point(ax, point["sensitivity"], point["ppv"], "operating point")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall (sensitivity)", ylabel="Precision (PPV)")
    _title(ax, "Precision-recall curve (test)", f"AUPRC {_fmt(point['auprc'])}")
    _save(fig, out_dir / "pr.png")


def _operating_point(ax, x: float, y: float, label: str) -> None:
    ax.scatter([x], [y], s=64, color=SERIES_2, edgecolors=SURFACE, linewidths=2, zorder=3)
    near_right = x > 0.7  # keep the label inside the axes
    ax.annotate(
        f"{label}\n({x:.2f}, {y:.2f})",
        (x, y),
        xytext=(-10 if near_right else 10, -26),
        textcoords="offset points",
        ha="right" if near_right else "left",
        color=TEXT_2,
        fontsize=8,
    )


def plot_reliability(
    y: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    bins: int,
    temperature: float,
    out_dir: Path,
) -> None:
    fig = _figure(5.2, 5.6)
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)
    ax = fig.add_subplot(grid[0])
    counts_ax = fig.add_subplot(grid[1], sharex=ax)
    _style(ax)
    _style(counts_ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.05, 0.12, "perfect calibration", color=MUTED, fontsize=8, rotation=37)
    series = (
        (raw, SERIES_1, "uncalibrated", "o"),
        (calibrated, SERIES_2, f"temperature-scaled (T={temperature:.2f})", "s"),
    )
    for probs, color, label, marker in series:
        mean_pred, observed, _ = reliability_curve(y, probs, bins)
        ece = expected_calibration_error(y, probs, bins)
        ax.plot(
            mean_pred,
            observed,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=6,
            markeredgecolor=SURFACE,
            label=f"{label}, ECE {ece:.3f}",
        )
    legend = fig.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=8, frameon=False, ncol=1
    )
    for text in legend.get_texts():
        text.set_color(TEXT_2)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), ylabel="Observed pneumonia rate")
    ax.tick_params(labelbottom=False)
    edges = np.linspace(0, 1, bins + 1)
    hist, _ = np.histogram(calibrated, bins=edges)
    counts_ax.bar(
        edges[:-1], hist, width=np.diff(edges) * 0.85, align="edge", color=SERIES_2, alpha=0.8
    )
    counts_ax.set(xlabel="Predicted P(pneumonia)", ylabel="Images")
    _title(ax, "Reliability diagram (test)", f"{bins} equal-width bins; histogram: calibrated")
    _save(fig, out_dir / "reliability.png")


def plot_gradcam_examples(
    predictor: Predictor,
    dataset,
    y: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    out_dir: Path,
    per_category: int = 2,
) -> None:
    """Grid of the most confident TP / TN / FP / FN cases with Grad-CAM for the predicted class."""
    predicted = probs >= threshold
    categories = {
        "TP": np.flatnonzero((y == 1) & predicted)[np.argsort(-probs[(y == 1) & predicted])],
        "TN": np.flatnonzero((y == 0) & ~predicted)[np.argsort(probs[(y == 0) & ~predicted])],
        "FP": np.flatnonzero((y == 0) & predicted)[np.argsort(-probs[(y == 0) & predicted])],
        "FN": np.flatnonzero((y == 1) & ~predicted)[np.argsort(probs[(y == 1) & ~predicted])],
    }
    fig = _figure(2.3 * len(categories), 2.5 * per_category + 0.6)
    axes = fig.subplots(per_category, len(categories), squeeze=False)
    for col, (name, indices) in enumerate(categories.items()):
        for row in range(per_category):
            ax = axes[row][col]
            ax.set_axis_off()
            if row >= len(indices):
                ax.text(0.5, 0.5, f"no {name}", ha="center", va="center", color=MUTED, fontsize=9)
                continue
            index = int(indices[row])
            image = raw_image(dataset, index)
            cam = predictor.explain(image)
            ax.imshow(overlay(predictor.model_view(image), cam, alpha=0.4))
            ax.set_title(f"{name} · P(pneumonia) {probs[index]:.2f}", color=TEXT, fontsize=9)
    fig.suptitle(
        "Grad-CAM for the predicted class (most confident cases per outcome). "
        "Coarse; not a localization method.",
        color=TEXT_2,
        fontsize=8,
    )
    _save(fig, out_dir / "gradcam_examples.png")


def _fmt(value: float) -> str:
    return f"{value:.3f}" if value is not None and math.isfinite(value) else "n/a"


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
