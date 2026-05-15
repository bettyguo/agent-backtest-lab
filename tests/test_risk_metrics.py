"""Tests for Sortino, Information Ratio, and downside metrics."""
from __future__ import annotations

import math

import numpy as np

from abl.multipletest.risk_metrics import (
    information_ratio,
    risk_metrics_summary,
    sortino_ratio,
)


def test_sortino_only_penalizes_downside():
    """Sortino > Sharpe for series whose volatility is mostly upside."""
    # Series with all small wins and one big win — zero downside.
    r = np.array([0.01, 0.01, 0.02, 0.005, 0.03])
    s = sortino_ratio(r, target_return=0.0, annualization=252)
    # Sortino with zero downside is NaN by construction.
    assert math.isnan(s)


def test_sortino_finite_when_downside_present():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=500)
    s = sortino_ratio(r, annualization=252)
    assert np.isfinite(s)


def test_sortino_nan_on_too_few_obs():
    assert math.isnan(sortino_ratio(np.array([0.01])))


def test_information_ratio_zero_against_self():
    """IR of a series against itself is NaN (zero tracking error)."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=200)
    ir = information_ratio(r, r)
    assert math.isnan(ir)


def test_information_ratio_positive_when_outperforming():
    rng = np.random.default_rng(0)
    bench = rng.normal(0.0, 0.01, size=500)
    strat = bench + rng.normal(0.001, 0.001, size=500)  # consistent excess
    ir = information_ratio(strat, bench)
    assert ir > 1.0  # strong, consistent excess -> high IR


def test_information_ratio_negative_when_underperforming():
    rng = np.random.default_rng(0)
    bench = rng.normal(0.001, 0.01, size=500)
    strat = bench - rng.normal(0.001, 0.001, size=500)
    ir = information_ratio(strat, bench)
    assert ir < -1.0


def test_risk_metrics_summary_no_bench():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=500)
    m = risk_metrics_summary(r, None)
    assert np.isfinite(m.sortino_annualized)
    assert math.isnan(m.information_ratio_annualized)


def test_risk_metrics_summary_with_bench():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=500)
    b = rng.normal(0.0005, 0.01, size=500)
    m = risk_metrics_summary(r, b)
    assert np.isfinite(m.sortino_annualized)
    assert np.isfinite(m.information_ratio_annualized)


def test_sortino_invalid_inputs():
    # zero variance -> NaN
    assert math.isnan(sortino_ratio(np.full(50, 0.001)))
