"""The honest scorecard.

A scorecard renders a BacktestResult (or several) into a Markdown + JSON report. Every
emitted report:

- shows net-of-cost returns only (gross is internal),
- shows confidence intervals on Sharpe,
- shows multiple-testing-corrected DSR when n_trials > 1,
- shows buy-and-hold and naive baselines alongside the evaluated adapter,
- shows explicit leakage and overfitting flags,
- carries the not-advice disclaimer.

There is no flag to suppress these.
"""
from __future__ import annotations

from abl.scorecard.breakdown import (
    TickerBreakdownRow,
    cross_strategy_correlation,
    effective_n_trials,
    per_ticker_breakdown,
)
from abl.scorecard.drawdown import DrawdownStats, drawdown_stats
from abl.scorecard.flags import UniverseFlag
from abl.scorecard.html_render import render_html
from abl.scorecard.report import (
    BaselineRow,
    Scorecard,
    build_scorecard,
    render_json,
    render_markdown,
)

__all__ = [
    "Scorecard",
    "BaselineRow",
    "UniverseFlag",
    "DrawdownStats",
    "drawdown_stats",
    "TickerBreakdownRow",
    "per_ticker_breakdown",
    "cross_strategy_correlation",
    "effective_n_trials",
    "build_scorecard",
    "render_markdown",
    "render_json",
    "render_html",
]
