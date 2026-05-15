"""Tests for PSR and DSR — known-answer fixtures."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from abl.multipletest.dsr import deflated_sharpe, expected_max_sharpe_under_null
from abl.multipletest.psr import probabilistic_sharpe, sharpe_ratio


def test_sharpe_known_answer():
    """Constant-vol Gaussian returns with µ=0.001, σ=0.01 → daily Sharpe 0.1.

    With N=10000 IID samples, the Sharpe estimator has SE ~ √(1/N) ≈ 0.01, so we
    allow ±0.02 absolute tolerance (2σ).
    """
    rng = np.random.default_rng(42)
    r = rng.normal(0.001, 0.01, size=10000)
    s = sharpe_ratio(r, annualization=252)
    assert s.sharpe_observed == pytest.approx(0.1, abs=0.02)
    assert s.sharpe_annualized == pytest.approx(0.1 * np.sqrt(252), abs=0.02 * np.sqrt(252))


def test_psr_zero_skew_zero_kurt_closed_form():
    """For Gaussian-like inputs (skew~0, excess kurt~0), PSR(0) reduces to Φ( ŜR · √(T-1) ).
    With ŜR = 0.1 and T = 1000, that's Φ(0.1 * √999) ≈ Φ(3.162) ≈ 0.99922.
    """
    rng = np.random.default_rng(42)
    r = rng.normal(0.001, 0.01, size=1000)
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    s = sharpe_ratio(r, annualization=252)
    expected = norm.cdf(
        s.sharpe_observed * np.sqrt(s.n_obs - 1)
        / np.sqrt(1.0 - s.skewness * s.sharpe_observed + (s.excess_kurtosis / 4.0) * s.sharpe_observed**2)
    )
    assert psr == pytest.approx(expected, rel=1e-9)


def test_dsr_reduces_to_psr_when_one_trial():
    """For n_trials=1 and var_trial_sharpe=0, SR_0 = 0, so DSR == PSR(0)."""
    rng = np.random.default_rng(42)
    r = rng.normal(0.0005, 0.01, size=1000)
    dsr = deflated_sharpe(r, n_trials=1, var_trial_sharpe=0.0)
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    assert dsr["dsr"] == pytest.approx(psr, rel=1e-9)
    assert dsr["sr_0"] == 0.0


def test_dsr_lower_than_psr_when_many_trials():
    """When n_trials >> 1 and var_trial_sharpe > 0, SR_0 > 0 so DSR < PSR(0)."""
    rng = np.random.default_rng(42)
    r = rng.normal(0.0005, 0.01, size=1000)
    psr = probabilistic_sharpe(r, sr_benchmark=0.0)
    dsr = deflated_sharpe(r, n_trials=100, var_trial_sharpe=0.5)
    assert dsr["sr_0"] > 0.0
    assert dsr["dsr"] < psr


def test_expected_max_monotone_in_n():
    v = 1.0
    vals = [expected_max_sharpe_under_null(n, v) for n in (1, 2, 5, 10, 100, 1000)]
    # First entry is 0 (n=1 special case); the rest must be strictly increasing.
    assert vals[0] == 0.0
    for a, b in zip(vals[1:-1], vals[2:], strict=True):
        assert b > a


def test_expected_max_zero_when_var_zero():
    assert expected_max_sharpe_under_null(100, 0.0) == 0.0


def test_dsr_full_output_shape():
    r = np.random.default_rng(0).normal(0.0005, 0.01, size=500)
    out = deflated_sharpe(r, n_trials=20, var_trial_sharpe=0.3)
    assert set(out.keys()) >= {"dsr", "sr_0", "sharpe_observed", "sharpe_annualized",
                                "n_trials", "var_trial_sharpe"}
    assert 0.0 <= out["dsr"] <= 1.0
