"""Tests for drawdown metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from abl.scorecard.drawdown import drawdown_stats


def test_no_drawdown_when_all_positive():
    r = np.array([0.01, 0.02, 0.01, 0.005])
    s = drawdown_stats(r, annualization=252)
    assert s.max_drawdown == 0.0
    assert s.longest_underwater_days == 0
    # No drawdown -> Calmar is NaN
    assert math.isnan(s.calmar_ratio)


def test_simple_drawdown_hand_computed():
    """Hand-computed: returns [+10%, -50%, +20%].
    Equity: 1.10, 0.55, 0.66.
    Peaks:  1.10, 1.10, 1.10.
    DD:      0%,  -50%, -40%.
    Max DD = -50%. Longest underwater = 2 days (days 2 and 3)."""
    r = np.array([0.10, -0.50, 0.20])
    s = drawdown_stats(r, annualization=252)
    assert s.max_drawdown == pytest.approx(-0.50, rel=1e-9)
    assert s.longest_underwater_days == 2
    assert s.final_equity == pytest.approx(1.10 * 0.50 * 1.20, rel=1e-9)


def test_drawdown_index():
    r = np.array([0.10, 0.10, -0.20, -0.10, 0.05])
    s = drawdown_stats(r)
    # Trough is at index 3 (cumulative product: 1.10, 1.21, 0.968, 0.8712, 0.91476)
    # Peak so far at each step: 1.10, 1.21, 1.21, 1.21, 1.21
    # DDs: 0, 0, -0.20, -0.28, -0.244
    assert s.max_drawdown_idx == 3
    assert s.max_drawdown == pytest.approx(-0.28, abs=1e-9)


def test_calmar_ratio_basic():
    """Annualized return = (final_equity)^(252/n) - 1; Calmar = ann / |max DD|.
    With n=252 and final_equity = 1.10, ann_ret = 0.10."""
    r = np.concatenate([np.full(126, 0.001), [-0.05], np.full(125, 0.001)])
    s = drawdown_stats(r, annualization=252)
    # Sanity: there IS a drawdown, ann ret is roughly the cum ret minus the loss
    assert s.max_drawdown < 0
    assert np.isfinite(s.calmar_ratio)


def test_drawdown_empty_series():
    s = drawdown_stats(np.array([]))
    assert s.max_drawdown == 0.0
    assert s.longest_underwater_days == 0
