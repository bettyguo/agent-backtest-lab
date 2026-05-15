"""The CI gate: the deliberately-leaky strategy MUST be caught.

This is the most important test in the repository. If this ever passes silently — i.e.
the leakage detector misses the leak — the build fails and the library is broken.

We construct a synthetic strategy that secretly reads the close of as_of+1 (the future)
from the raw store, bypassing the PITView firewall. This is intentional in test code and
NEVER in product code. The test asserts that the post-hoc detector flags impossibly-high
directional accuracy.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from abl.adapters.callable_adapter import CallableAdapter
from abl.backtest.engine import WalkForwardEngine
from abl.baselines.naive_momentum import naive_momentum_adapter
from abl.data.loaders import DataStore
from abl.leakage.detector import detect_leakage
from abl.types import Decision, PITViewProtocol


def _make_leaky_adapter(store: DataStore) -> CallableAdapter:
    """Build a strategy that secretly sees tomorrow. NEVER do this in product code."""

    def leaky_fn(ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        raw = store.raw_bars(ticker)
        # Locate the next trading day after as_of and copy its sign.
        future = raw[raw.index > pd.Timestamp(as_of)]
        if future.empty:
            return Decision(direction="FLAT", confidence=None)
        next_close = float(future.iloc[0]["close"])
        # Find today's close (the legitimate observation).
        today = raw[raw.index == pd.Timestamp(as_of)]
        if today.empty:
            return Decision(direction="FLAT", confidence=None)
        today_close = float(today.iloc[0]["close"])
        direction = "LONG" if next_close > today_close else "SHORT"
        return Decision(direction=direction, confidence=None, raw={"leaky": True})

    return CallableAdapter(fn=leaky_fn, name="LEAKY_TEST_STRATEGY")


def test_leaky_strategy_is_caught(synth_store, synth_universe, synth_window):
    """The leakage detector MUST flag the impossible accuracy of the leaky strategy.

    This is the gate test. Do not loosen the assertion. If a refactor breaks this,
    the firewall or the detector is broken, and that is the fact this test is built to
    surface.
    """
    leaky = _make_leaky_adapter(synth_store)
    engine = WalkForwardEngine(store=synth_store)
    result = engine.run(leaky, synth_universe, synth_window)
    report = detect_leakage(audit_events=result.audit_events, decisions=result.decisions)
    codes = [f.code for f in report.flags]
    assert (
        "IMPOSSIBLE_ACCURACY" in codes
    ), f"Leakage detector failed to catch the deliberately-leaky strategy. Flags: {codes}"
    # And severity is critical.
    critical = [f for f in report.flags if f.code == "IMPOSSIBLE_ACCURACY"]
    assert critical[0].severity == "critical"


def test_clean_strategy_is_not_flagged(synth_store, synth_universe, synth_window):
    """A clean baseline MUST NOT trigger any leakage flag.

    This is the false-positive gate: a benign strategy passing through the firewall
    cleanly must produce an empty leakage report.
    """
    clean = naive_momentum_adapter(lookback=20)
    engine = WalkForwardEngine(store=synth_store)
    result = engine.run(clean, synth_universe, synth_window)
    report = detect_leakage(audit_events=result.audit_events, decisions=result.decisions)
    assert not report.any, f"Leakage detector raised false positive on naive momentum: {report.flags}"


def test_firewall_audit_log_records_all_accesses(synth_store, synth_universe, synth_window):
    """The audit log should record one event per PITView call from the adapter."""
    naive = naive_momentum_adapter(lookback=20)
    engine = WalkForwardEngine(store=synth_store)
    result = engine.run(naive, synth_universe, synth_window)
    # Naive momentum calls pit.bars exactly once per ticker per day.
    n_days = len(result.daily_pnl_net)
    n_tickers = len(synth_universe.tickers)
    # Allow some slack for edge days
    assert len(result.audit_events) >= n_days * n_tickers - 5
    # Zero violations expected in a clean run.
    violations = [e for e in result.audit_events if not e["allowed"]]
    assert violations == []
