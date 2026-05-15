"""Naive mean-reversion baseline: LONG if trailing return is negative, FLAT if positive.

The mirror of `naive_momentum`. Useful when comparing an agent against both directional
priors so a "this agent is just trend-following" or "this agent is just contrarian"
hypothesis is easy to falsify.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from abl.adapters.plain_strategy import PlainStrategyAdapter
from abl.types import Decision, PITViewProtocol


@dataclass
class _NaiveMeanReversion:
    lookback: int = 5

    def __call__(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        bars = pit_data.bars(ticker, lookback_days=self.lookback + 1)
        if bars is None or bars.empty or len(bars) < 2:
            return Decision(direction="FLAT", confidence=None,
                            raw={"baseline": "naive_mean_reversion"})
        first_close = float(bars.iloc[0]["close"])
        last_close = float(bars.iloc[-1]["close"])
        # LONG if trailing return is NEGATIVE (expect bounce); FLAT otherwise.
        direction = "LONG" if last_close < first_close else "FLAT"
        return Decision(direction=direction, confidence=None,
                        raw={"baseline": "naive_mean_reversion", "lookback": self.lookback})


def naive_mean_reversion_adapter(lookback: int = 5) -> PlainStrategyAdapter:
    return PlainStrategyAdapter(
        fn=_NaiveMeanReversion(lookback=lookback),
        name=f"naive_mean_reversion_{lookback}",
    )
