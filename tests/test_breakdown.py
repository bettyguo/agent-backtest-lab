"""Tests for per-ticker breakdown + cross-strategy correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abl.scorecard.breakdown import (
    cross_strategy_correlation,
    effective_n_trials,
    per_ticker_breakdown,
)


def _make_decisions_df():
    """Three tickers × 5 days. A is right 100%, B is right 0%, C is FLAT every day."""
    rows = []
    for d_str, retA, retB in (
        ("2020-01-02", +0.01, -0.01),
        ("2020-01-03", -0.01, +0.01),
        ("2020-01-06", +0.02, -0.02),
        ("2020-01-07", -0.02, +0.02),
        ("2020-01-08", +0.005, -0.005),
    ):
        ts = pd.Timestamp(d_str)
        rows.append({
            "as_of": ts, "ticker": "A",
            "direction": "LONG" if retA > 0 else "SHORT",
            "confidence": 0.9, "position": 1.0, "ret_next": retA,
        })
        rows.append({
            "as_of": ts, "ticker": "B",
            "direction": "LONG" if retB < 0 else "SHORT",  # always wrong
            "confidence": 0.9, "position": 1.0, "ret_next": retB,
        })
        rows.append({
            "as_of": ts, "ticker": "C",
            "direction": "FLAT", "confidence": None, "position": 0.0, "ret_next": 0.001,
        })
    return pd.DataFrame(rows)


def test_per_ticker_breakdown_hit_rates():
    df = _make_decisions_df()
    rows = per_ticker_breakdown(df, annualization=252)
    by_ticker = {r.ticker: r for r in rows}
    assert by_ticker["A"].hit_rate == pytest.approx(1.0)
    assert by_ticker["B"].hit_rate == pytest.approx(0.0)
    # C is all FLAT so non-FLAT count is 0 -> hit_rate is NaN
    assert np.isnan(by_ticker["C"].hit_rate)


def test_per_ticker_breakdown_counts():
    df = _make_decisions_df()
    rows = per_ticker_breakdown(df)
    by_ticker = {r.ticker: r for r in rows}
    assert by_ticker["A"].n_decisions == 5
    assert by_ticker["A"].n_non_flat == 5
    assert by_ticker["C"].n_non_flat == 0


def test_per_ticker_breakdown_empty():
    assert per_ticker_breakdown(pd.DataFrame()) == []


def test_cross_strategy_correlation_identity():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    s1 = pd.Series(rng.standard_normal(100), index=idx)
    s2 = s1.copy()
    corr = cross_strategy_correlation({"a": s1, "b": s2})
    assert corr.loc["a", "b"] == pytest.approx(1.0)
    assert corr.loc["a", "a"] == pytest.approx(1.0)


def test_cross_strategy_correlation_independent():
    idx = pd.date_range("2020-01-01", periods=10000, freq="B")
    rng = np.random.default_rng(0)
    s1 = pd.Series(rng.standard_normal(10000), index=idx)
    s2 = pd.Series(rng.standard_normal(10000), index=idx)
    corr = cross_strategy_correlation({"a": s1, "b": s2})
    assert abs(corr.loc["a", "b"]) < 0.05  # near zero on large independent samples


def test_effective_n_trials_iid_recovers_N():
    """Identity correlation matrix -> effective N = N."""
    M = pd.DataFrame(np.eye(10), columns=list("abcdefghij"), index=list("abcdefghij"))
    assert effective_n_trials(M) == pytest.approx(10.0, rel=1e-9)


def test_effective_n_trials_collapses_when_correlated():
    """All-ones correlation -> effective N = 1."""
    M = pd.DataFrame(np.ones((5, 5)), columns=list("abcde"), index=list("abcde"))
    n_eff = effective_n_trials(M)
    assert n_eff == pytest.approx(1.0, abs=0.01)


def test_effective_n_trials_intermediate():
    """Block-correlated (two blocks of 5) -> effective N near 2."""
    M = np.eye(10) * 1.0
    # Two blocks of strong intra-correlation
    for i in range(5):
        for j in range(5):
            M[i, j] = 1.0 if i == j else 0.95
            M[i + 5, j + 5] = 1.0 if i == j else 0.95
    df = pd.DataFrame(M, columns=range(10), index=range(10))
    n_eff = effective_n_trials(df)
    assert 1.5 <= n_eff <= 3.0, f"expected ~2 effective trials, got {n_eff}"


def test_effective_n_trials_empty():
    assert effective_n_trials(pd.DataFrame()) == 0.0
