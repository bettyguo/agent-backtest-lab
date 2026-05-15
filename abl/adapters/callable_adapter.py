"""Generic callable adapter — wrap any user function in the StrategyAdapter protocol."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from abl.types import Decision, PITViewProtocol


@dataclass
class CallableAdapter:
    """Wrap any `Callable[[ticker, as_of, pit_data], Decision]` as a StrategyAdapter.

    This is the escape hatch for any framework that doesn't have a dedicated adapter.
    """

    fn: Callable[[str, date, PITViewProtocol], Decision]
    name: str = "callable"

    def predict(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        return self.fn(ticker, as_of, pit_data)
