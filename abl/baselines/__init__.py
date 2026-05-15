"""Built-in baselines — buy-and-hold, naive momentum, random.

The scorecard ALWAYS shows the evaluated adapter alongside all three baselines. There
is no flag to suppress this. Comparing to a passive baseline is the single cheapest
way to expose a backtest result that looked impressive but underperformed doing nothing.
"""
from __future__ import annotations

from abl.baselines.buy_and_hold import buy_and_hold_adapter
from abl.baselines.dca import dca_adapter
from abl.baselines.equal_vol import equal_vol_adapter
from abl.baselines.mean_reversion import naive_mean_reversion_adapter
from abl.baselines.naive_momentum import naive_momentum_adapter
from abl.baselines.random_baseline import random_baseline_adapter

__all__ = [
    "buy_and_hold_adapter",
    "naive_momentum_adapter",
    "random_baseline_adapter",
    "equal_vol_adapter",
    "naive_mean_reversion_adapter",
    "dca_adapter",
]
