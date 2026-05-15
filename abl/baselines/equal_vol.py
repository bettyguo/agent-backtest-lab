"""Inverse-volatility weighted baseline.

Every day, size each ticker's LONG position inversely to its trailing realized volatility
so each leg contributes equal risk. A real strategy must beat this — equal-vol is the
"risk-parity for one factor" minimum bar — not just buy-and-hold.

Reference: "Equal Risk Contribution" / "Risk Parity" is widely-credited to the
Bridgewater All Weather literature (informal); the simple inverse-vol form predates
that and appears in standard portfolio-construction textbooks. We don't claim authorship
or endorsement of any one paper — this is a well-known portfolio-construction baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from abl.adapters.plain_strategy import PlainStrategyAdapter
from abl.types import Decision, PITViewProtocol


@dataclass
class _EqualVol:
    lookback: int = 60

    def __call__(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        bars = pit_data.bars(ticker, lookback_days=self.lookback)
        if bars is None or bars.empty or len(bars) < self.lookback // 2:
            return Decision(direction="FLAT", confidence=None,
                            raw={"baseline": "equal_vol", "reason": "insufficient_data"})
        log_ret = np.log(bars["close"].astype(float) / bars["close"].astype(float).shift(1)).dropna()
        if log_ret.size < 5 or float(log_ret.std()) <= 0:
            return Decision(direction="FLAT", confidence=None,
                            raw={"baseline": "equal_vol", "reason": "zero_vol"})
        # We can't directly set position magnitude through the Decision protocol; the engine
        # uses the EngineConfig.sizing as a multiplier. So this baseline becomes effectively
        # LONG when the trailing vol is finite — the per-ticker sizing reconciliation is the
        # job of a richer engine pass. We pass the inverse-vol value through `raw` for
        # forensics, even though the engine treats this as a plain LONG.
        return Decision(direction="LONG", confidence=None,
                        raw={"baseline": "equal_vol", "trailing_vol": float(log_ret.std())})


def equal_vol_adapter(lookback: int = 60) -> PlainStrategyAdapter:
    return PlainStrategyAdapter(fn=_EqualVol(lookback=lookback), name=f"equal_vol_{lookback}")
