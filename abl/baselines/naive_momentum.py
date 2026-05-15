"""Naive momentum baseline: LONG if the trailing window had positive return, else FLAT.

This is a stupid strategy on purpose. It exists to give the scorecard a non-passive
baseline that nonetheless has zero look-ahead capability, so the user can see that
"a useful strategy must beat both buy-and-hold AND a stupid trend-follower, after costs."
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from abl.adapters.plain_strategy import PlainStrategyAdapter
from abl.types import Decision, PITViewProtocol


@dataclass
class _NaiveMomentum:
    lookback: int = 20

    def __call__(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        bars = pit_data.bars(ticker, lookback_days=self.lookback)
        if bars is None or bars.empty or len(bars) < 2:
            return Decision(direction="FLAT", confidence=None, raw={"baseline": "naive_momentum"})
        first_close = float(bars.iloc[0]["close"])
        last_close = float(bars.iloc[-1]["close"])
        if last_close > first_close:
            direction = "LONG"
        else:
            direction = "FLAT"
        return Decision(direction=direction, confidence=None,
                        raw={"baseline": "naive_momentum", "lookback": self.lookback})


def naive_momentum_adapter(lookback: int = 20) -> PlainStrategyAdapter:
    return PlainStrategyAdapter(fn=_NaiveMomentum(lookback=lookback), name=f"naive_momentum_{lookback}")
