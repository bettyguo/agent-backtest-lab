"""Example 1: evaluate a plain (non-LLM) strategy end-to-end on the bundled fixture.

Run:
    python examples/01_evaluate_plain_strategy.py

What this does:
1. Loads the bundled synthetic fixture (no API keys needed).
2. Wraps a tiny mean-reversion rule as a CallableAdapter.
3. Runs the walk-forward engine, the three baselines, and builds an honest scorecard.
4. Writes report.md and report.json into ./abl_example_out/.

This is the simplest possible end-to-end demonstration of the library. Read it before
attempting Example 2 (which integrates TauricResearch/TradingAgents).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from abl.adapters.callable_adapter import CallableAdapter
from abl.backtest.engine import WalkForwardEngine
from abl.baselines import buy_and_hold_adapter, naive_momentum_adapter, random_baseline_adapter
from abl.data.loaders import load_fixture
from abl.scorecard.report import build_scorecard, render_json, render_markdown
from abl.types import Decision, Universe, Window


def my_mean_reversion(ticker, as_of, pit_data) -> Decision:
    """Trivial mean-reversion rule. LONG if today's close < trailing 5-day mean. FLAT otherwise.

    This is illustrative only — it would not survive transaction costs on real data, which
    is exactly the kind of result the scorecard is built to surface honestly.
    """
    bars = pit_data.bars(ticker, lookback_days=6)
    if bars.empty or len(bars) < 6:
        return Decision(direction="FLAT")
    last_close = float(bars.iloc[-1]["close"])
    trailing_mean = float(bars.iloc[:-1]["close"].mean())
    if last_close < trailing_mean:
        return Decision(direction="LONG", confidence=0.55)
    return Decision(direction="FLAT")


def main() -> None:
    print("This is a research and evaluation tool. Not financial advice. Not a trading system.")
    store = load_fixture()
    universe = Universe(
        tickers=store.tickers(),
        name="bundled_synthetic",
        survivorship_verified=True,
        source_note="synthetic GBM; no survivorship bias by construction",
    )
    window = Window(start=date(2020, 1, 6), end=date(2024, 6, 28))

    adapter = CallableAdapter(fn=my_mean_reversion, name="example_mean_reversion")
    engine = WalkForwardEngine(store=store)
    primary = engine.run(adapter, universe, window)
    baselines = [
        engine.run(buy_and_hold_adapter(), universe, window),
        engine.run(naive_momentum_adapter(), universe, window),
        engine.run(random_baseline_adapter(), universe, window),
    ]
    sc = build_scorecard(primary=primary, baselines=baselines)
    out = Path("abl_example_out")
    out.mkdir(exist_ok=True)
    (out / "report.md").write_text(render_markdown(sc), encoding="utf-8")
    (out / "report.json").write_text(render_json(sc), encoding="utf-8")
    print(f"Scorecard written to {out / 'report.md'}")


if __name__ == "__main__":
    main()
