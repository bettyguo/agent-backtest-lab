"""Adapter for AI4Finance-Foundation/FinRobot.

Like FinGPT, FinRobot is a framework with many entry points rather than a single
"decide direction" call. We provide a thin adapter that wraps a user-supplied callable
which returns a `(direction_str, confidence_or_none)` pair. The adapter normalizes the
direction string and validates the confidence.

The `finrobot` framework is intentionally an OPTIONAL dependency; the adapter works
without it installed because the user supplies the callable.

References
----------
- AI4Finance-Foundation/FinRobot — https://github.com/AI4Finance-Foundation/FinRobot
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from abl.adapters.tradingagents import _default_direction_parser
from abl.types import Decision, PITViewProtocol


@dataclass
class FinRobotAdapter:
    """Wrap a user-supplied (ticker, as_of, pit_data) -> (text_or_label, conf) callable.

    The text-or-label part is parsed through the same defensive parser used for
    TradingAgents — it looks for LONG/SHORT/FLAT keywords and returns FLAT as the
    safe fallback. Custom parsers are passable via `direction_parser`.
    """

    decision_fn: Callable[[str, date, PITViewProtocol], tuple[object, float | None]]
    direction_parser: Callable[[object], str] = _default_direction_parser
    name: str = "finrobot"

    def predict(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        raw_decision, conf = self.decision_fn(ticker, as_of, pit_data)
        direction = self.direction_parser(raw_decision)
        if conf is not None:
            try:
                cf = float(conf)
                if not (0.0 <= cf <= 1.0):
                    cf = None
            except (TypeError, ValueError):
                cf = None
        else:
            cf = None
        return Decision(direction=direction, confidence=cf, raw={"finrobot_raw": raw_decision})
