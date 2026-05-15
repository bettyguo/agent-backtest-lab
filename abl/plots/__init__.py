"""Matplotlib plotters for scorecard artifacts.

These produce PNGs that the scorecard Markdown can reference. We use matplotlib's Agg
backend so plots can be generated headlessly (no display required). No seaborn / no
heavy plotting deps — pure matplotlib.

Available
---------
- `plot_equity_curve(...)`: cumulative net equity vs each baseline.
- `plot_drawdown(...)`: drawdown curve with max-DD trough annotated.
- `plot_reliability_diagram(...)`: confidence vs empirical accuracy by bin.
"""
from __future__ import annotations

from abl.plots.renderers import (
    plot_drawdown,
    plot_equity_curve,
    plot_reliability_diagram,
)

__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_reliability_diagram",
]
