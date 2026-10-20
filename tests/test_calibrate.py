import numpy as np
import pytest

from lungxray.calibrate import fit_temperature, softmax_np


def _sample(true_temperature: float, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 3.0, size=(n, 1))
    logits = np.hstack([-z, z])
    p_positive = softmax_np(logits, true_temperature)[:, 1]
    labels = (rng.random(n) < p_positive).astype(int)
    return logits, labels


@pytest.mark.parametrize("true_temperature", [0.5, 1.0, 2.5])
def test_fit_temperature_recovers_known_temperature(true_temperature):
    logits, labels = _sample(true_temperature)
    assert fit_temperature(logits, labels) == pytest.approx(true_temperature, rel=0.15)


def test_softmax_rows_sum_to_one():
    probs = softmax_np(np.array([[2.0, -1.0], [0.0, 0.0]]), 2.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[1, 0] == pytest.approx(0.5)
