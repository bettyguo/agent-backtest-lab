"""Tests for transaction-cost models — hand-computed known answers."""
from __future__ import annotations

import pandas as pd
import pytest

from abl.costs.models import AlmgrenImpactCost, CompositeCost, ConstantBpsCost


def test_constant_bps_hand_computed():
    """For 5 bps each side and turnover [1.0, 0.5, 0.0, 2.0] the cost is
    [5e-4, 2.5e-4, 0.0, 1e-3]."""
    turnover = pd.Series([1.0, 0.5, 0.0, 2.0], index=pd.date_range("2020-01-01", periods=4))
    cost = ConstantBpsCost(bps_each_side=5.0).cost_fraction(turnover)
    assert cost.iloc[0] == pytest.approx(5e-4)
    assert cost.iloc[1] == pytest.approx(2.5e-4)
    assert cost.iloc[2] == pytest.approx(0.0)
    assert cost.iloc[3] == pytest.approx(1e-3)


def test_almgren_known_answer():
    """eta=1, beta=0.5, sigma=0.01, participation=0.04 -> impact 0.01*0.2 = 2e-3.
    With turnover 1.0 the cost is 1.0 * 2e-3 = 2e-3."""
    turnover = pd.Series([1.0], index=pd.date_range("2020-01-01", periods=1))
    cost = AlmgrenImpactCost(eta=1.0, beta=0.5, sigma_default=0.01).cost_fraction(
        turnover, context={"participation": pd.Series([0.04], index=turnover.index)}
    )
    assert cost.iloc[0] == pytest.approx(2e-3, rel=1e-6)


def test_almgren_with_three_fifths_exponent():
    """The Almgren-Thum-Hauptmann-Li 2005 estimate is beta=3/5. Verify the formula
    plumbs that through."""

    turnover = pd.Series([1.0], index=pd.date_range("2020-01-01", periods=1))
    cost = AlmgrenImpactCost(eta=1.0, beta=0.6, sigma_default=0.01).cost_fraction(
        turnover, context={"participation": pd.Series([0.04], index=turnover.index)}
    )
    expected = 0.01 * (0.04 ** 0.6)
    assert cost.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_composite_cost_sums():
    turnover = pd.Series([1.0], index=pd.date_range("2020-01-01", periods=1))
    m1 = ConstantBpsCost(bps_each_side=5.0)
    m2 = ConstantBpsCost(bps_each_side=3.0)
    c = CompositeCost(models=(m1, m2)).cost_fraction(turnover)
    assert c.iloc[0] == pytest.approx(8e-4)
