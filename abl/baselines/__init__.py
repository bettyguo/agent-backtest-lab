"""Built-in baselines — buy-and-hold, naive momentum, random.

The scorecard ALWAYS shows the evaluated adapter alongside all three baselines. There
is no flag to suppress this. Comparing to a passive baseline is the single cheapest
way to expose a backtest result that looked impressive but underperformed doing nothing.
"""
from __future__ import annotations

from abl.baselines.buy_and_hold import buy_and_hold_adapter
from abl.baselines.naive_momentum import naive_momentum_adapter
from abl.baselines.random_baseline import random_baseline_adapter

__all__ = ["buy_and_hold_adapter", "naive_momentum_adapter", "random_baseline_adapter"]
