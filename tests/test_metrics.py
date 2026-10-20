import math

import numpy as np
import pytest

from lungxray.metrics import (
    METRIC_NAMES,
    binary_metrics,
    bootstrap_ci,
    confusion_counts,
    expected_calibration_error,
    reliability_curve,
    threshold_for_sensitivity,
    youden_threshold,
)

LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
PROBS = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1, 0.1, 0.05])


def test_binary_metrics_known_confusion():
    assert confusion_counts(LABELS, PROBS, 0.5) == {"tn": 5, "fp": 1, "fn": 1, "tp": 3}
    m = binary_metrics(LABELS, PROBS, 0.5)
    assert set(m) == set(METRIC_NAMES)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(5 / 6)
    assert m["ppv"] == pytest.approx(0.75)
    assert m["npv"] == pytest.approx(5 / 6)
    assert m["f1"] == pytest.approx(0.75)
    assert 0.5 < m["auroc"] <= 1.0 and 0.0 < m["auprc"] <= 1.0


def test_threshold_is_inclusive():
    assert binary_metrics(LABELS, PROBS, 0.7)["sensitivity"] == pytest.approx(0.75)


def test_single_class_metrics_are_nan_not_errors():
    m = binary_metrics(np.ones(4), np.array([0.2, 0.4, 0.6, 0.8]), 0.5)
    assert math.isnan(m["auroc"]) and math.isnan(m["specificity"])
    assert m["sensitivity"] == pytest.approx(0.5)


def test_threshold_for_sensitivity_is_the_highest_reaching_target():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    probs = np.clip(rng.normal(0.35 + 0.3 * labels, 0.2), 0, 1)
    t = threshold_for_sensitivity(labels, probs, 0.95)
    assert binary_metrics(labels, probs, t)["sensitivity"] >= 0.95
    higher = np.unique(probs[probs > t])[0]
    assert binary_metrics(labels, probs, higher)["sensitivity"] < 0.95
    assert threshold_for_sensitivity(labels, probs, 1.0) == probs[labels == 1].min()
    with pytest.raises(ValueError, match="positive"):
        threshold_for_sensitivity(np.zeros(3), np.ones(3), 0.9)


def test_youden_threshold_separates_perfectly_separable_scores():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = youden_threshold(labels, probs)
    m = binary_metrics(labels, probs, t)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_ece_values():
    labels = np.array([1] * 3 + [0] * 7)
    assert expected_calibration_error(labels, np.full(10, 0.3)) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([1, 0] * 5), np.full(10, 0.9)) == pytest.approx(0.4)
    mean_pred, observed, counts = reliability_curve(labels, np.full(10, 0.3), 15)
    assert counts.tolist() == [10] and observed[0] == pytest.approx(0.3)


def test_bootstrap_ci_is_seeded_and_brackets_point():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 300)
    probs = np.clip(rng.normal(0.3 + 0.4 * labels, 0.2), 0, 1)
    a = bootstrap_ci(labels, probs, 0.5, n_resamples=200, seed=7)
    b = bootstrap_ci(labels, probs, 0.5, n_resamples=200, seed=7)
    assert a == b and set(a) == set(METRIC_NAMES)
    point = binary_metrics(labels, probs, 0.5)
    for name in ("accuracy", "auroc", "sensitivity", "specificity"):
        low, high = a[name]
        assert low <= point[name] <= high


def test_bootstrap_ci_none_when_never_defined():
    ci = bootstrap_ci(np.ones(5), np.full(5, 0.7), 0.5, n_resamples=20)
    assert ci["auroc"] is None and ci["specificity"] is None and ci["sensitivity"] == (1.0, 1.0)
