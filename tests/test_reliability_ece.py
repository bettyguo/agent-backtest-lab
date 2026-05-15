"""Tests for reliability diagrams and ECE."""
from __future__ import annotations

import numpy as np
import pytest

from abl.calibration.reliability import (
    expected_calibration_error,
    reliability_diagram_data,
)


def test_perfectly_calibrated_synthetic():
    """probs ~ Uniform(0, 1), labels ~ Bernoulli(probs). ECE -> 0 as N -> infinity."""
    rng = np.random.default_rng(20260514)
    n = 50000
    probs = rng.uniform(0, 1, size=n)
    labels = (rng.uniform(0, 1, size=n) < probs).astype(int)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    # With N=50000 and 10 bins, the per-bin SE is small; ECE should be well under 0.02.
    assert ece < 0.02


def test_uncalibrated_constant_high_confidence():
    """Always predict p=0.95 but the truth is 50/50 -> ECE = 0.45."""
    n = 10000
    probs = np.full(n, 0.95)
    labels = np.random.default_rng(0).integers(0, 2, size=n)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.45, abs=0.02)


def test_reliability_bins_omit_empty():
    probs = np.array([0.05, 0.95])
    labels = np.array([0, 1])
    bins = reliability_diagram_data(probs, labels, n_bins=10)
    assert len(bins) == 2  # only the [0, 0.1) and [0.9, 1] bins are populated


def test_reliability_invalid_inputs():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5]), np.array([0, 1]))
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([-0.1, 0.5]), np.array([0, 1]))
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5, 0.6]), np.array([0, 2]))
