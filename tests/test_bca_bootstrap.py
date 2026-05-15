"""Tests for the BCa bootstrap (Efron 1987)."""
from __future__ import annotations

import numpy as np
import pytest

from abl.multipletest.bootstrap import bca_bootstrap_ci, bca_sharpe_ci


def test_bca_normal_mean_covers_true():
    """Empirical 95% coverage of BCa CI for the mean on Gaussian samples should be near 95%."""
    rng = np.random.default_rng(20260514)
    n_rep = 200
    n = 100
    mu = 0.5
    sigma = 1.0
    covered = 0
    for _ in range(n_rep):
        x = rng.normal(mu, sigma, size=n)
        res = bca_bootstrap_ci(x, statistic_fn=lambda a: float(a.mean()),
                               n_bootstrap=600, alpha=0.05, rng=rng)
        if res.ci_low <= mu <= res.ci_high:
            covered += 1
    cov = covered / n_rep
    assert cov >= 0.88, f"empirical coverage {cov:.3f} far below nominal 0.95"


def test_bca_sharpe_ci_contains_estimate():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, size=300)
    res = bca_sharpe_ci(r, n_bootstrap=500, alpha=0.05, rng=rng)
    assert res.ci_low <= res.statistic <= res.ci_high


def test_bca_does_not_crash_on_degenerate_input():
    """The BCa wrapper must not crash on a near-constant series — degenerate but
    realistic input. The reported statistic and CI may be uninformative; we don't
    constrain their values here, only that the function returns."""
    rng = np.random.default_rng(0)
    r = np.full(100, 0.001)  # near-constant; floating-point noise gives tiny variance
    res = bca_sharpe_ci(r, n_bootstrap=100, rng=rng)
    # Function returned without raising — that's the property under test.
    assert isinstance(res.statistic, float)


def test_bca_invalid_alpha():
    with pytest.raises(ValueError):
        bca_bootstrap_ci(np.arange(20.0), statistic_fn=float.__call__, alpha=1.5)


def test_bca_too_few_observations():
    with pytest.raises(ValueError):
        bca_bootstrap_ci(np.array([1.0, 2.0]), statistic_fn=lambda a: float(a.mean()))


def test_bca_acceleration_finite():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    res = bca_bootstrap_ci(x, statistic_fn=lambda a: float(a.mean()),
                           n_bootstrap=400, rng=rng)
    assert np.isfinite(res.acceleration)
    assert np.isfinite(res.z_0)
