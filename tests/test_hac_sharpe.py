"""Tests for the HAC / Newey-West Sharpe SE (Lo 2002)."""
from __future__ import annotations

import numpy as np
import pytest

from abl.multipletest.hac import (
    auto_lag_truncation,
    hac_sharpe_ci,
    newey_west_eta,
)


def test_auto_lag_truncation_known_values():
    # floor(4 * (T/100)^(2/9)) for a few sizes
    assert auto_lag_truncation(100) == 4  # floor(4*1) = 4
    assert auto_lag_truncation(252) >= 4
    assert auto_lag_truncation(5) == 0  # too small


def test_eta_reduces_to_one_when_no_autocorr():
    """η = 1 + 2·Σ w_k·ρ_k. With ρ_k=0 it must be exactly 1."""
    rhos = np.zeros(5)
    assert newey_west_eta(rhos) == 1.0


def test_eta_positive_with_positive_autocorr():
    """Positive autocorrelations -> η > 1."""
    rhos = np.array([0.5, 0.3, 0.1])
    eta = newey_west_eta(rhos)
    # Bartlett weights for q=3: (3/4, 2/4, 1/4)
    expected = 1.0 + 2.0 * (0.75 * 0.5 + 0.5 * 0.3 + 0.25 * 0.1)
    assert eta == pytest.approx(expected, rel=1e-10)


def test_hac_reduces_to_iid_when_q_zero():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=500)
    res = hac_sharpe_ci(r, q_lags=0)
    assert res.eta_q == 1.0
    assert res.se_hac == pytest.approx(res.se_iid, rel=1e-10)


def test_hac_widens_ci_under_positive_autocorr():
    """An AR(1) process with positive φ has positive autocorrelation; HAC SE > IID SE."""
    rng = np.random.default_rng(20260514)
    T = 1000
    phi = 0.5
    eps = rng.normal(0, 0.01, size=T)
    r = np.empty(T)
    r[0] = 0.0
    for t in range(1, T):
        r[t] = phi * r[t - 1] + eps[t]
    # Center around a small positive drift so Sharpe is well-defined
    r = r + 0.0005
    res = hac_sharpe_ci(r)
    assert res.eta_q > 1.0, f"expected eta > 1 for positive autocorr, got {res.eta_q}"
    assert res.se_hac > res.se_iid


def test_hac_returns_finite_ci_on_white_noise():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=1000)
    res = hac_sharpe_ci(r, annualization=252)
    assert np.isfinite(res.ci_low_obs)
    assert np.isfinite(res.ci_high_obs)
    assert np.isfinite(res.ci_low_ann)
    assert np.isfinite(res.ci_high_ann)
    assert res.ci_low_obs < res.sharpe_observed < res.ci_high_obs


def test_hac_invalid_inputs():
    with pytest.raises(ValueError):
        hac_sharpe_ci(np.array([0.01]))  # only 1 obs
    with pytest.raises(ValueError):
        hac_sharpe_ci(np.zeros(10))  # zero variance
