"""PlainStrategyAdapter — same callable interface, used by the built-in baselines.

The point of having a separate class is purely to make scorecard provenance clearer:
when the scorecard prints "baseline: buy_and_hold", it reads `adapter.name` and there
is no ambiguity about whether the adapter is a user strategy or a built-in baseline.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from abl.types import Decision, PITViewProtocol


@dataclass
class PlainStrategyAdapter:
    """Lightweight wrapper for the built-in baselines."""

    fn: Callable[[str, date, PITViewProtocol], Decision]
    name: str

    def predict(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        return self.fn(ticker, as_of, pit_data)
