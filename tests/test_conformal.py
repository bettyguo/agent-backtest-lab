"""Tests for split conformal prediction."""
from __future__ import annotations

import numpy as np

from abl.calibration.conformal import (
    rolling_split_conformal_coverage,
    split_conformal_quantile,
)


def test_quantile_index():
    """For n=10 calibration scores [1..10] and alpha=0.1, ⌈(11)(0.9)⌉/10 = ⌈9.9⌉/10 = 10/10.
    So q = sorted[9] = 10."""
    cal = np.arange(1, 11, dtype=float)
    q = split_conformal_quantile(cal, alpha=0.1)
    assert q == 10.0


def test_quantile_index_alpha_05():
    """For n=20 and alpha=0.05, k=⌈21*0.95⌉=⌈19.95⌉=20 → q = max of cal."""
    cal = np.arange(1, 21, dtype=float)
    q = split_conformal_quantile(cal, alpha=0.05)
    assert q == 20.0


def test_split_conformal_empirical_coverage_iid():
    """Empirical coverage of split conformal on IID exchangeable data should >= 1 - alpha.

    Repeat 500 times: draw 100 calibration scores from a standard normal, draw one test
    score from the same distribution, compute prediction quantile, check coverage.
    """
    rng = np.random.default_rng(20260514)
    alpha = 0.1
    n_rep = 500
    covered = 0
    for _ in range(n_rep):
        cal = rng.standard_normal(100)
        test = rng.standard_normal()
        q = split_conformal_quantile(cal, alpha=alpha)
        if test <= q:
            covered += 1
    emp_cov = covered / n_rep
    # Marginal coverage ≥ 1 - alpha (0.9). SE for binomial proportion with n=500 around 0.9
    # is √(0.9*0.1/500) ≈ 0.013. Allow 2σ slack on the lower side.
    assert emp_cov >= 0.9 - 0.03, f"empirical coverage {emp_cov:.3f} < 0.9 - 2σ"


def test_rolling_split_conformal_returns_expected_keys():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(500)
    out = rolling_split_conformal_coverage(scores, alpha=0.1, window=100)
    assert set(out.keys()) == {"alpha_nominal", "empirical_coverage", "n_eval", "median_quantile"}
    assert out["alpha_nominal"] == 0.1
    assert out["n_eval"] == 400


def test_rolling_split_conformal_iid_coverage():
    """On IID exchangeable scores the rolling-window coverage should hit ~0.9 at alpha=0.1."""
    rng = np.random.default_rng(20260514)
    scores = rng.standard_normal(2000)
    out = rolling_split_conformal_coverage(scores, alpha=0.1, window=200)
    assert 0.85 <= out["empirical_coverage"] <= 0.95
