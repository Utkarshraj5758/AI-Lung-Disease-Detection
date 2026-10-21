"""Write the results sections of README.md and docs/MODEL_CARD.md from reports/metrics.json.

    python scripts/render_results.py [--check]

Only the text between ``<!-- BEGIN GENERATED: name -->`` and ``<!-- END GENERATED: name -->``
is replaced. ``--check`` changes nothing and exits 1 if a file is out of date; the test suite
runs it, so every number in the docs comes from ``python -m lungxray.evaluate``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics.json"
TARGETS = {ROOT / "README.md": ("results",), ROOT / "docs" / "MODEL_CARD.md": ("model-card",)}
TBD = "TBD (run `make evaluate`)."
ROWS = (  # (key, label, as percentage)
    ("auroc", "AUROC", False),
    ("auprc", "AUPRC", False),
    ("accuracy", "Accuracy", True),
    ("sensitivity", "Sensitivity (recall)", True),
    ("specificity", "Specificity", True),
    ("ppv", "PPV (precision)", True),
    ("npv", "NPV", True),
    ("f1", "F1", False),
    ("ece", "ECE", False),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a file is out of date")
    parser.add_argument("--metrics", type=Path, default=METRICS)
    args = parser.parse_args(argv)
    report = load_report(args.metrics)
    blocks = {"results": readme_results(report), "model-card": model_card_results(report)}
    stale = []
    for path, names in TARGETS.items():
        raw = path.read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n")
        updated = text
        for name in names:
            updated = replace_block(updated, name, blocks[name])
        if updated != text:
            stale.append(path)
            if not args.check:
                newline = "\r\n" if b"\r\n" in raw else "\n"
                path.write_bytes(updated.replace("\n", newline).encode("utf-8"))
    for path in stale:
        verb = "is out of date" if args.check else "updated"
        print(f"{path.relative_to(ROOT)} {verb}")
    return 1 if args.check and stale else 0


def load_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("fast_dev_run"):
        raise SystemExit(f"{path} comes from --fast-dev-run; refusing to publish it")
    return report


def replace_block(text: str, name: str, body: str) -> str:
    tag = re.escape(name)
    pattern = re.compile(
        rf"(<!-- BEGIN GENERATED: {tag} -->\n).*?(<!-- END GENERATED: {tag} -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(f"Missing <!-- BEGIN/END GENERATED: {name} --> markers")
    return pattern.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(2), text)


def fmt(entry: dict[str, Any] | None, percent: bool) -> str:
    """``0.988 (0.976-0.996)`` or ``92.7% (90.5-94.7)``; ``n/a`` when undefined."""
    if not entry or entry.get("value") is None:
        return "n/a"
    scale, digits, unit = (100, 1, "%") if percent else (1, 3, "")
    text = f"{entry['value'] * scale:.{digits}f}{unit}"
    if entry.get("ci95"):
        low, high = (v * scale for v in entry["ci95"])
        text += f" ({low:.{digits}f}–{high:.{digits}f})"
    return text


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _metrics_table(report: dict[str, Any]) -> list[str]:
    n, unseen = report["n"], report.get("test_unseen_patients")
    header = f"| Metric | All test images (n = {n['total']}) |"
    rule = "| --- | --- |"
    if unseen:
        header += f" Test patients not in train (n = {unseen['n']['total']}) |"
        rule += " --- |"
    lines = [header, rule]
    for key, label, percent in ROWS:
        row = f"| {label} | {fmt(report['metrics'][key], percent)} |"
        if unseen:
            row += f" {fmt(unseen['metrics'][key], percent)} |"
        lines.append(row)
    return lines


def _setup_sentence(report: dict[str, Any]) -> str:
    n, boot = report["n"], report["bootstrap"]
    return (
        f"Test split: **{n['total']} images** ({n['NORMAL']} NORMAL, {n['PNEUMONIA']} "
        f"PNEUMONIA). Decision threshold **{report['threshold']:.3f}** on temperature-scaled "
        f"P(pneumonia) (T = {report['temperature']:.2f}); both were fitted on the validation "
        f"split, never on test. 95% CIs in parentheses: percentile bootstrap, "
        f"{boot['resamples']} resamples, seed {boot['seed']}. ECE uses "
        f"{report['ece_definition']}."
    )


def _overlap_note(report: dict[str, Any], limitations: str) -> str | None:
    overlap = (report.get("patient_overlap") or {}).get("train/test")
    unseen = report.get("test_unseen_patients")
    if not overlap or not unseen:
        return None
    u = unseen["n"]
    return (
        f"> **Patient overlap.** {overlap['test_images_from_shared_patients']} of the "
        f"{overlap['test_images']} test images come from the {overlap['shared_patients']} "
        "patients who also appear in the training split (the dataset folders were used as "
        f"provided; see [Limitations]({limitations})), so the left column is optimistic. The "
        f"right column keeps only test patients never seen in training ({u['NORMAL']} NORMAL, "
        f"{u['PNEUMONIA']} PNEUMONIA); its class mix differs, so PPV and NPV are not "
        "comparable across the columns."
    )


def _confusion_sentence(report: dict[str, Any]) -> str:
    c = report["confusion_matrix"]
    return (
        f"Confusion matrix (all test images): {c['tp']} true positives, {c['fn']} false "
        f"negatives (missed pneumonia), {c['fp']} false positives, {c['tn']} true negatives."
    )


def _validation_sentence(report: dict[str, Any]) -> str | None:
    val = report.get("validation")
    if not val:
        return None
    return (
        f"Validation split ({val['n']} images; used only for model selection, the temperature "
        f"and the threshold): the threshold is the highest reaching "
        f"{val['target_sensitivity']:.0%} sensitivity there ({_pct(val['sensitivity'])}, "
        f"specificity {_pct(val['specificity'])}; Youden-J threshold "
        f"{val['youden_threshold']:.3f}). Validation AUROC {val['auroc']:.3f}."
    )


def _calibration_sentence(report: dict[str, Any]) -> str:
    val = report.get("validation") or {}
    text = (
        f"Temperature scaling (T = {report['temperature']:.2f}) changed test ECE from "
        f"{report['ece_uncalibrated']:.3f} to {report['metrics']['ece']['value']:.3f}"
    )
    if val:
        text += (
            f"; on validation it lowered NLL from {val['nll_before']:.3f} to "
            f"{val['nll_after']:.3f} while ECE went from {val['ece_before']:.3f} to "
            f"{val['ece_after']:.3f}"
        )
    return text + "."


def _provenance(report: dict[str, Any]) -> str:
    model = report["model"]
    trained = (model.get("git_commit") or "unknown")[:7]
    evaluated = (report.get("evaluation_git_commit") or "unknown")[:7]
    return (
        f"<sub>Generated by `scripts/render_results.py` from `reports/metrics.json`: "
        f"{model['model_name']} trained at commit {trained}, evaluated at commit {evaluated} on "
        f"{report['evaluated_at'][:10]}, dataset fingerprint "
        f"`{report.get('dataset_fingerprint') or model.get('dataset_fingerprint')}`.</sub>"
    )


def readme_results(report: dict[str, Any] | None) -> str:
    if report is None:
        return TBD
    parts = [_setup_sentence(report), "\n".join(_metrics_table(report))]
    parts += [p for p in (_overlap_note(report, "#limitations"),) if p]
    parts.append(_confusion_sentence(report))
    parts += [p for p in (_validation_sentence(report),) if p]
    parts.append(_calibration_sentence(report))
    parts.append(
        '<p align="center">\n'
        '  <img src="reports/confusion_matrix.png" alt="Confusion matrix" width="32%">\n'
        '  <img src="reports/roc.png" alt="ROC curve" width="32%">\n'
        '  <img src="reports/reliability.png" alt="Reliability diagram" width="32%">\n'
        "</p>"
    )
    parts.append(_provenance(report))
    return "\n\n".join(parts)


def model_card_results(report: dict[str, Any] | None) -> str:
    if report is None:
        return TBD
    parts = [
        "### Test-set metrics",
        _setup_sentence(report),
        "\n".join(_metrics_table(report)),
    ]
    parts += [p for p in (_overlap_note(report, "#caveats-and-limitations"),) if p]
    parts.append(_confusion_sentence(report))
    parts.append("### Validation and operating point")
    parts += [p for p in (_validation_sentence(report),) if p]
    parts.append("### Calibration")
    parts.append(
        _calibration_sentence(report)
        + " Temperature scaling minimizes NLL, not ECE, so treat the displayed probability as "
        "approximate. See `reports/reliability.png`."
    )
    parts.append(_provenance(report))
    return "\n\n".join(parts)


if __name__ == "__main__":
    sys.exit(main())
