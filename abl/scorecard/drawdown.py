"""Drawdown metrics: max drawdown, longest underwater stretch, Calmar ratio.

Drawdown is the single metric most users intuitively understand and the one academic
backtest research routinely under-emphasizes. We compute it from the net-of-cost equity
curve only.

Definitions
-----------
- Equity curve: cumulative product of (1 + r_t) starting at 1.0.
- Running peak: cummax of the equity curve.
- Drawdown_t: (equity_t / peak_t) - 1. Always in [-1, 0].
- Max drawdown: min over t of drawdown_t.
- Longest underwater stretch: max contiguous run of t with drawdown_t < 0.
- Calmar ratio: annualized return / |max drawdown|. The classic "pain-adjusted return"
  metric, originally Young (1991) "Calmar Ratio: A Smoother Tool."

These are NOT statistical guarantees — they describe the realized path, not the
distribution. We report them alongside Sharpe / DSR so users can see both "how
much risk-adjusted return" and "how much rope to drawdown."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abl.config import DEFAULT_ANNUALIZATION


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float          # in [-1, 0]
    max_drawdown_idx: int        # index at which the trough occurred
    longest_underwater_days: int  # longest contiguous span with drawdown < 0
    calmar_ratio: float          # annualized return / |max drawdown|
    final_equity: float          # last value of the equity curve


def drawdown_stats(
    returns: np.ndarray, *, annualization: int = DEFAULT_ANNUALIZATION
) -> DrawdownStats:
    """Compute drawdown statistics from a per-period return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 1:
        return DrawdownStats(
            max_drawdown=0.0,
            max_drawdown_idx=-1,
            longest_underwater_days=0,
            calmar_ratio=float("nan"),
            final_equity=1.0,
        )
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    md_idx = int(np.argmin(dd))
    md = float(dd[md_idx])

    # Longest underwater stretch
    underwater = dd < 0
    longest = 0
    cur = 0
    for u in underwater:
        if u:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    final_eq = float(equity[-1])
    n_days = r.size
    if n_days > 0 and md < 0:
        ann_ret = float((final_eq) ** (annualization / n_days) - 1.0)
        calmar = ann_ret / abs(md)
    else:
        calmar = float("nan")

    return DrawdownStats(
        max_drawdown=md,
        max_drawdown_idx=md_idx,
        longest_underwater_days=longest,
        calmar_ratio=calmar,
        final_equity=final_eq,
    )
