"""Example 2: evaluate TauricResearch/TradingAgents.

NOT RUN IN CI. This example requires:
  1. The `tradingagents` framework installed (https://github.com/TauricResearch/TradingAgents).
  2. API keys for an LLM provider (OpenAI, etc.) configured per that framework's docs.
  3. Real point-in-time-correct market data (the bundled synthetic fixture is fine for the
     adapter call mechanics, but the LLM will not produce realistic decisions against it).

This file is intentionally illustrative: it documents the integration shape so users can
adapt it to their own setup. Running it on a real universe will incur API costs proportional
to (n_tickers × n_trading_days × cost_per_LLM_call). Caveat user.

What this does:
1. Wraps TradingAgentsGraph().propagate() with TradingAgentsAdapter.
2. Runs the same walk-forward engine and baselines as Example 1.
3. Builds a scorecard with multiple-testing correction (DSR) because users almost always
   try several prompt/config variants — and lying about that defeats the correction.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from abl.backtest.engine import WalkForwardEngine
from abl.baselines import buy_and_hold_adapter, naive_momentum_adapter, random_baseline_adapter
from abl.data.loaders import load_fixture
from abl.scorecard.report import build_scorecard, render_json, render_markdown
from abl.types import Universe, Window


def main() -> None:
    print("This is a research and evaluation tool. Not financial advice. Not a trading system.")
    print("This example requires the TauricResearch/TradingAgents framework installed.")
    print("Visit https://github.com/TauricResearch/TradingAgents")
    print()
    try:
        from abl.adapters.tradingagents import TradingAgentsAdapter
        adapter = TradingAgentsAdapter(
            ta_config={
                # TradingAgents config — see their docs. Common keys: llm_provider, deep_think_llm,
                # quick_think_llm, max_debate_rounds. Pinning a cheap model for cost control:
                "llm_provider": "openai",
                "deep_think_llm": "gpt-4o-mini",
                "quick_think_llm": "gpt-4o-mini",
                "max_debate_rounds": 1,
            },
            debug=False,
        )
    except ImportError as e:
        print(f"[abl] TradingAgentsAdapter is not usable: {e}")
        print("[abl] Install TradingAgents and re-run.")
        return

    store = load_fixture()  # see file docstring — replace with your real PIT data
    universe = Universe(
        tickers=("SYN-A",),
        name="bundled_synthetic_one_ticker",
        survivorship_verified=True,
    )
    window = Window(start=date(2024, 1, 2), end=date(2024, 2, 1))  # short window — LLM calls are expensive

    engine = WalkForwardEngine(store=store)
    primary = engine.run(adapter, universe, window)
    baselines = [
        engine.run(buy_and_hold_adapter(), universe, window),
        engine.run(naive_momentum_adapter(), universe, window),
        engine.run(random_baseline_adapter(), universe, window),
    ]
    # Be honest about how many configurations you tried — N=5 here as a placeholder.
    sc = build_scorecard(
        primary=primary,
        baselines=baselines,
        n_trials_reported=5,
        var_trial_sharpe=0.3,
    )
    out = Path("abl_tradingagents_out")
    out.mkdir(exist_ok=True)
    (out / "report.md").write_text(render_markdown(sc), encoding="utf-8")
    (out / "report.json").write_text(render_json(sc), encoding="utf-8")
    print(f"Scorecard written to {out / 'report.md'}")


if __name__ == "__main__":
    main()
