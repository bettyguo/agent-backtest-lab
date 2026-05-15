"""Dollar-cost-averaging baseline: gradually scales into a LONG position.

For the first `ramp_days` of the run, position scales linearly from 0 to full LONG;
afterward fully LONG. This baseline captures the "I'd have just averaged in" retail
counterfactual, which is a non-trivial bar to clear for a strategy whose backtest happens
to start at a market trough.

We can't directly modulate position sizing through the Decision protocol, so the engine
treats the LONG decisions identically once the ramp ends. The ramp phase emits FLAT
until the index reaches `ramp_days`. This is a minor approximation; the scorecard
notes the simplification in the adapter `raw` payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from abl.adapters.plain_strategy import PlainStrategyAdapter
from abl.types import Decision, PITViewProtocol


@dataclass
class _DCA:
    ramp_days: int = 20
    _seen: dict = field(default_factory=dict)

    def __call__(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        seen = self._seen.setdefault(ticker, 0)
        self._seen[ticker] = seen + 1
        if seen < self.ramp_days:
            return Decision(direction="FLAT", confidence=None,
                            raw={"baseline": "dca", "ramping": True, "day": seen})
        return Decision(direction="LONG", confidence=None,
                        raw={"baseline": "dca", "day": seen})


def dca_adapter(ramp_days: int = 20) -> PlainStrategyAdapter:
    return PlainStrategyAdapter(fn=_DCA(ramp_days=ramp_days), name=f"dca_{ramp_days}")
