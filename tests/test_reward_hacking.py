"""Tests for reward-hacking / window-overfitting detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from abl.leakage.reward_hacking import detect_reward_hacking


def test_no_flag_when_consistent():
    """Same distribution across the window — no reward-hacking flag."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    r = rng.normal(0.0005, 0.01, size=500)
    flags = detect_reward_hacking(returns=pd.Series(r, index=idx), is_fraction=0.7)
    assert len(flags) == 0


def test_critical_flag_when_sharpe_collapses():
    """High Sharpe IS, near-zero / negative Sharpe OOS — critical flag."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    # First 350 days: strong positive drift; remaining 150: pure noise.
    is_r = rng.normal(0.003, 0.01, size=350)
    oos_r = rng.normal(-0.0005, 0.012, size=150)
    series = pd.Series(np.concatenate([is_r, oos_r]), index=idx)
    flags = detect_reward_hacking(returns=series, is_fraction=0.7)
    codes = [f.code for f in flags]
    assert "SHARPE_DROP_IS_OOS" in codes
    crit = [f for f in flags if f.code == "SHARPE_DROP_IS_OOS"]
    assert crit[0].severity == "critical"


def test_drawdown_widens_oos_flag():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    # IS: tiny drawdowns. OOS: a big drawdown injected.
    is_r = rng.normal(0.0008, 0.005, size=350)
    oos_r = rng.normal(-0.001, 0.02, size=150)
    oos_r[60:80] = -0.04  # forced big drawdown
    series = pd.Series(np.concatenate([is_r, oos_r]), index=idx)
    flags = detect_reward_hacking(returns=series, is_fraction=0.7)
    codes = [f.code for f in flags]
    assert "DRAWDOWN_WIDENS_OOS" in codes or "SHARPE_DROP_IS_OOS" in codes


def test_too_short_returns_empty():
    """Series shorter than the minimum should return no flags rather than raise."""
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    series = pd.Series(np.zeros(20), index=idx)
    assert detect_reward_hacking(returns=series) == []


def test_calibration_divergence_flag():
    """Confidence well-calibrated in IS half, badly miscalibrated in OOS half."""
    rng = np.random.default_rng(20260514)
    n_is = 200
    n_oos = 200
    # IS: high confidence + high accuracy.
    rows = []
    base_date = pd.Timestamp("2020-01-01")
    for i in range(n_is):
        rows.append({
            "as_of": base_date + pd.Timedelta(days=i),
            "ticker": "X",
            "direction": "LONG",
            "confidence": 0.9,
            "position": 1.0,
            "ret_next": 0.01 if rng.random() < 0.9 else -0.01,  # ~90% correct
        })
    # OOS: high confidence but only ~50% correct.
    for i in range(n_oos):
        rows.append({
            "as_of": base_date + pd.Timedelta(days=n_is + i),
            "ticker": "X",
            "direction": "LONG",
            "confidence": 0.9,
            "position": 1.0,
            "ret_next": 0.01 if rng.random() < 0.5 else -0.01,
        })
    decisions = pd.DataFrame(rows)
    idx = pd.DatetimeIndex(decisions["as_of"])
    returns_series = pd.Series(
        decisions["position"].astype(float).to_numpy() * decisions["ret_next"].astype(float).to_numpy(),
        index=idx,
    )
    flags = detect_reward_hacking(returns=returns_series, decisions=decisions, is_fraction=0.5)
    codes = [f.code for f in flags]
    assert "CALIBRATION_DIVERGES_OOS" in codes
