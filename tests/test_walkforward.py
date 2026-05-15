"""Walk-forward engine + purged-CV tests."""
from __future__ import annotations

import pandas as pd

from abl.backtest.cv import purged_kfold_splits
from abl.backtest.engine import EngineConfig, WalkForwardEngine
from abl.baselines.buy_and_hold import buy_and_hold_adapter
from abl.costs.models import ConstantBpsCost


def test_purged_kfold_purges_label_window():
    """No training index lies within [test_start - label_window, test_end + label_window + embargo)."""
    n = 1000
    k = 5
    label = 7
    embargo_pct = 0.02
    embargo = int(round(embargo_pct * n))
    splits = purged_kfold_splits(n, k, label_window=label, embargo_pct=embargo_pct)
    assert len(splits) == k
    for s in splits:
        test_start, test_end = int(s.test_idx[0]), int(s.test_idx[-1]) + 1
        purge_lo = max(0, test_start - label)
        purge_hi = min(n, test_end + label + embargo)
        # No training index can sit inside the purge window
        assert ((s.train_idx >= purge_hi) | (s.train_idx < purge_lo)).all()


def test_purged_kfold_train_test_disjoint():
    splits = purged_kfold_splits(500, 4, label_window=3, embargo_pct=0.01)
    for s in splits:
        assert set(s.train_idx).isdisjoint(set(s.test_idx))


def test_walkforward_buy_and_hold_is_deterministic(synth_store, synth_universe, synth_window):
    engine1 = WalkForwardEngine(store=synth_store)
    engine2 = WalkForwardEngine(store=synth_store)
    bah = buy_and_hold_adapter()
    r1 = engine1.run(bah, synth_universe, synth_window)
    r2 = engine2.run(bah, synth_universe, synth_window)
    pd.testing.assert_series_equal(r1.daily_pnl_net, r2.daily_pnl_net)


def test_walkforward_net_lt_gross_when_costs_applied(synth_store, synth_universe, synth_window):
    """With a non-zero cost model and an adapter that turns over each day, net < gross."""
    from abl.adapters.callable_adapter import CallableAdapter
    from abl.types import Decision

    flip = {"x": True}

    def flipping(ticker, as_of, pit_data):
        flip["x"] = not flip["x"]
        return Decision(direction="LONG" if flip["x"] else "SHORT")

    engine = WalkForwardEngine(
        store=synth_store, config=EngineConfig(cost_model=ConstantBpsCost(bps_each_side=10.0))
    )
    result = engine.run(CallableAdapter(fn=flipping, name="flipper"), synth_universe, synth_window)
    cumulative_gross = result.daily_pnl_gross.sum()
    cumulative_net = result.daily_pnl_net.sum()
    assert cumulative_net < cumulative_gross


def test_walkforward_window_too_small_raises(synth_store, synth_universe):
    from datetime import date

    import pytest

    from abl.types import Window

    tiny = Window(start=date(2020, 1, 6), end=date(2020, 1, 6))
    engine = WalkForwardEngine(store=synth_store)
    with pytest.raises(ValueError):
        engine.run(buy_and_hold_adapter(), synth_universe, tiny)
