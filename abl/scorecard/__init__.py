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

from abl.scorecard.flags import UniverseFlag
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
    "build_scorecard",
    "render_markdown",
    "render_json",
]
