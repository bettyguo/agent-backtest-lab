"""Buy-and-hold baseline: LONG every ticker, every day, full sizing, no confidence."""
from __future__ import annotations

from datetime import date

from abl.adapters.plain_strategy import PlainStrategyAdapter
from abl.types import Decision, PITViewProtocol


def _buy_and_hold(ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
    return Decision(direction="LONG", confidence=None, raw={"baseline": "buy_and_hold"})


def buy_and_hold_adapter() -> PlainStrategyAdapter:
    return PlainStrategyAdapter(fn=_buy_and_hold, name="buy_and_hold")
