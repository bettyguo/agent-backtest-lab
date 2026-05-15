"""Tests for the honest scorecard."""
from __future__ import annotations

import json

from abl.backtest.engine import WalkForwardEngine
from abl.baselines import buy_and_hold_adapter, naive_momentum_adapter, random_baseline_adapter
from abl.config import DISCLAIMER_LINE
from abl.scorecard.report import build_scorecard, render_json, render_markdown


def _run_with_baselines(store, universe, window):
    engine = WalkForwardEngine(store=store)
    primary = engine.run(buy_and_hold_adapter(), universe, window)
    baselines = [
        engine.run(naive_momentum_adapter(), universe, window),
        engine.run(random_baseline_adapter(), universe, window),
    ]
    return primary, baselines


def test_scorecard_basic_shape(synth_store, synth_universe, synth_window):
    primary, baselines = _run_with_baselines(synth_store, synth_universe, synth_window)
    sc = build_scorecard(primary=primary, baselines=baselines)
    assert sc.adapter_name == "buy_and_hold"
    assert sc.universe_size == 3
    assert len(sc.baselines) == 2
    assert 0.0 <= sc.psr <= 1.0


def test_scorecard_markdown_contains_disclaimer(synth_store, synth_universe, synth_window):
    primary, baselines = _run_with_baselines(synth_store, synth_universe, synth_window)
    sc = build_scorecard(primary=primary, baselines=baselines)
    md = render_markdown(sc)
    # Disclaimer line MUST appear.
    assert DISCLAIMER_LINE in md
    # Disclaimer block (first phrase) MUST appear.
    assert "This project is a research and evaluation tool" in md
    # Gross-return-only language MUST NOT leak in.
    assert "Gross total return" not in md
    assert "gross return" not in md.lower() or "net" in md.lower()


def test_scorecard_markdown_renders_baselines(synth_store, synth_universe, synth_window):
    primary, baselines = _run_with_baselines(synth_store, synth_universe, synth_window)
    sc = build_scorecard(primary=primary, baselines=baselines)
    md = render_markdown(sc)
    assert "naive_momentum" in md
    assert "random" in md


def test_scorecard_json_round_trip(synth_store, synth_universe, synth_window):
    primary, baselines = _run_with_baselines(synth_store, synth_universe, synth_window)
    sc = build_scorecard(primary=primary, baselines=baselines)
    js = render_json(sc)
    payload = json.loads(js)
    assert payload["adapter_name"] == "buy_and_hold"
    assert "disclaimer" in payload
    assert "leakage_flags" in payload
    assert isinstance(payload["leakage_flags"], list)


def test_scorecard_with_dsr_when_multiple_trials(synth_store, synth_universe, synth_window):
    primary, baselines = _run_with_baselines(synth_store, synth_universe, synth_window)
    sc = build_scorecard(
        primary=primary, baselines=baselines, n_trials_reported=50, var_trial_sharpe=0.4
    )
    assert sc.dsr is not None
    assert sc.sr_0 is not None and sc.sr_0 > 0
    md = render_markdown(sc)
    assert "DSR" in md


def test_scorecard_survivorship_flag_when_unverified(synth_store, synth_window):
    from abl.types import Universe
    unverified = Universe(tickers=("SYN-A", "SYN-B", "SYN-C"), name="unverified",
                          survivorship_verified=False)
    engine = WalkForwardEngine(store=synth_store)
    primary = engine.run(buy_and_hold_adapter(), unverified, synth_window)
    sc = build_scorecard(primary=primary, baselines=[])
    codes = [f.code for f in sc.universe_flags]
    assert "SURVIVORSHIP_UNVERIFIED" in codes


def test_scorecard_calibration_not_evaluable_when_no_confidence(synth_store, synth_universe, synth_window):
    primary, baselines = _run_with_baselines(synth_store, synth_universe, synth_window)
    sc = build_scorecard(primary=primary, baselines=baselines)
    # buy_and_hold emits no confidence
    assert sc.confidence_emitted is False
    assert sc.ece is None
    md = render_markdown(sc)
    assert "not evaluable" in md.lower() or "not provided" in md.lower()
