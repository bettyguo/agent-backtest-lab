"""Tests for the three adapters."""
from __future__ import annotations

import pytest

from abl.adapters.callable_adapter import CallableAdapter
from abl.adapters.plain_strategy import PlainStrategyAdapter
from abl.backtest.engine import WalkForwardEngine
from abl.baselines import buy_and_hold_adapter, naive_momentum_adapter, random_baseline_adapter
from abl.types import Decision


def test_callable_adapter_passes_through(synth_store, synth_universe, synth_window):
    calls = {"n": 0}

    def fn(ticker, as_of, pit_data):
        calls["n"] += 1
        return Decision(direction="LONG", confidence=0.6)

    adapter = CallableAdapter(fn=fn, name="test")
    engine = WalkForwardEngine(store=synth_store)
    result = engine.run(adapter, synth_universe, synth_window)
    assert calls["n"] > 0
    assert result.adapter_name == "test"
    assert (result.decisions["confidence"] == 0.6).all()


def test_plain_strategy_adapter_runs(synth_store, synth_universe, synth_window):
    adapter = buy_and_hold_adapter()
    assert isinstance(adapter, PlainStrategyAdapter)
    engine = WalkForwardEngine(store=synth_store)
    result = engine.run(adapter, synth_universe, synth_window)
    assert (result.decisions["direction"] == "LONG").all()


def test_three_baselines_run_end_to_end(synth_store, synth_universe, synth_window):
    engine = WalkForwardEngine(store=synth_store)
    for adapter in (buy_and_hold_adapter(), naive_momentum_adapter(), random_baseline_adapter()):
        result = engine.run(adapter, synth_universe, synth_window)
        assert len(result.daily_pnl_net) > 0


def test_tradingagents_adapter_clean_import_error():
    """TradingAgentsAdapter should raise a clean ImportError at construction when the
    framework is not installed."""
    from abl.adapters.tradingagents import TradingAgentsAdapter

    # The framework is not in our pyproject; instantiation should fail with ImportError.
    with pytest.raises(ImportError) as exc_info:
        TradingAgentsAdapter()
    assert "TradingAgents" in str(exc_info.value)


def test_decision_validates_bounds():
    with pytest.raises(ValueError):
        Decision(direction="LONG", confidence=1.5)
    with pytest.raises(ValueError):
        Decision(direction="BUY")  # type: ignore[arg-type]


def test_decision_accepts_none_confidence():
    d = Decision(direction="LONG", confidence=None)
    assert d.confidence is None
