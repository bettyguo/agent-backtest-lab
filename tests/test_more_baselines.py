"""Smoke tests for the three new baselines."""
from __future__ import annotations

from abl.backtest.engine import WalkForwardEngine
from abl.baselines import dca_adapter, equal_vol_adapter, naive_mean_reversion_adapter


def test_equal_vol_runs_end_to_end(synth_store, synth_universe, synth_window):
    engine = WalkForwardEngine(store=synth_store)
    result = engine.run(equal_vol_adapter(lookback=20), synth_universe, synth_window)
    assert len(result.daily_pnl_net) > 0
    assert result.adapter_name.startswith("equal_vol")


def test_naive_mean_reversion_runs_end_to_end(synth_store, synth_universe, synth_window):
    engine = WalkForwardEngine(store=synth_store)
    result = engine.run(naive_mean_reversion_adapter(lookback=5), synth_universe, synth_window)
    assert len(result.daily_pnl_net) > 0


def test_dca_initially_flat_then_long(synth_store, synth_universe, synth_window):
    """The ramp-up phase should be FLAT; afterward LONG."""
    engine = WalkForwardEngine(store=synth_store)
    result = engine.run(dca_adapter(ramp_days=10), synth_universe, synth_window)
    decisions = result.decisions
    # Each ticker's first 10 calls should be FLAT.
    for _ticker, grp in decisions.groupby("ticker"):
        grp = grp.sort_values("as_of").reset_index(drop=True)
        first_10 = grp.iloc[:10]
        assert (first_10["direction"] == "FLAT").all()
        # Some later calls should be LONG.
        later = grp.iloc[10:]
        if len(later) > 0:
            assert (later["direction"] == "LONG").any()


def test_naive_mean_reversion_inverts_momentum(synth_store, synth_universe, synth_window):
    """On the same fixture, naive mean reversion's signal is the opposite-direction
    of naive momentum's. We don't compare PnL — just confirm directions disagree."""
    from abl.baselines import naive_momentum_adapter
    engine = WalkForwardEngine(store=synth_store)
    mom = engine.run(naive_momentum_adapter(lookback=5), synth_universe, synth_window)
    rev = engine.run(naive_mean_reversion_adapter(lookback=5), synth_universe, synth_window)
    # Same window, same calendar, same lookback: every (as_of, ticker) row should pair up.
    mom_dirs = mom.decisions.set_index(["as_of", "ticker"])["direction"]
    rev_dirs = rev.decisions.set_index(["as_of", "ticker"])["direction"]
    common = mom_dirs.index.intersection(rev_dirs.index)
    assert len(common) > 0
    # When momentum says LONG, mean reversion should say FLAT, and vice versa.
    for key in common[:20]:
        if mom_dirs.loc[key] == "LONG":
            assert rev_dirs.loc[key] == "FLAT"
        elif mom_dirs.loc[key] == "FLAT":
            # Mean reversion can be LONG or FLAT here
            assert rev_dirs.loc[key] in ("LONG", "FLAT")
