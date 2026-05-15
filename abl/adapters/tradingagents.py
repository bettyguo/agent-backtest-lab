"""Adapter for TauricResearch/TradingAgents.

This is an OPTIONAL integration. We never import `tradingagents` at module load. The
adapter raises a clear ImportError at construction time if the framework is not installed.

The adapter:
1. Calls `TradingAgentsGraph(debug=False, config=ta_config).propagate(ticker, as_of.isoformat())`.
2. Parses the returned `decision` into our `Decision`.
3. Tries to extract a numeric confidence if a custom parser is supplied; otherwise leaves
   it as None. We do NOT fabricate a confidence — calibration reports "not evaluable"
   in that case. This is honest.

Important caveat: this adapter cannot prevent the TradingAgents framework from making
its own out-of-band data calls (e.g. via LLM tool use to a news API). The leakage firewall
mediates adapter-driven access through PITView, but the framework's internals are not
inspectable from here. The post-hoc leakage detectors (`abl.leakage`) are designed
precisely for that case. We document this loudly in the README and in the adapter docstring.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from abl.types import Decision, Direction, PITViewProtocol


def _default_direction_parser(raw: Any) -> Direction:
    """Best-effort string parser. Looks at common patterns in TradingAgents output.

    TradingAgents' README documents that `propagate` returns a 2-tuple whose second
    element is a "decision" — the exact schema is not documented in the README we
    inspected at design time and may change. This parser is intentionally defensive:
    it scans the string for unambiguous direction keywords. If nothing matches, returns
    FLAT — the safest fallback. Users who want different behavior should pass a custom
    `decision_parser`.
    """
    if raw is None:
        return "FLAT"
    s = str(raw).upper()
    # Order matters: check FLAT/HOLD before BUY/SELL substrings (e.g. "HOLD" should not match "OLD" in another word).
    if "HOLD" in s or "FLAT" in s or "NEUTRAL" in s or "NO POSITION" in s:
        return "FLAT"
    if "SHORT" in s or "SELL" in s:
        return "SHORT"
    if "LONG" in s or "BUY" in s:
        return "LONG"
    return "FLAT"


@dataclass
class TradingAgentsAdapter:
    """Wraps TauricResearch/TradingAgents `TradingAgentsGraph().propagate(ticker, date)`.

    Construct-time check: imports `tradingagents` lazily. If the framework is not
    installed, raises ImportError with an actionable hint.
    """

    ta_config: dict | None = None
    decision_parser: Callable[[Any], Direction] = _default_direction_parser
    confidence_parser: Callable[[Any], float | None] | None = None
    debug: bool = False
    name: str = "tradingagents"

    def __post_init__(self) -> None:
        try:
            from tradingagents.default_config import DEFAULT_CONFIG  # noqa: F401
            from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: F401
        except Exception as exc:  # ImportError or anything else surfaced by the framework
            raise ImportError(
                "TradingAgentsAdapter requires the TauricResearch/TradingAgents framework. "
                "Install it from https://github.com/TauricResearch/TradingAgents. This adapter "
                "is intentionally an optional dependency."
            ) from exc

        # Resolve config lazily; defer instantiation to first predict() so misconfiguration
        # is reported per-call rather than at __init__.
        from tradingagents.default_config import DEFAULT_CONFIG

        cfg = dict(DEFAULT_CONFIG)
        if self.ta_config is not None:
            cfg.update(self.ta_config)
        self._cfg = cfg
        self._graph = None  # built on first call

    def _ensure_graph(self):
        if self._graph is None:
            from tradingagents.graph.trading_graph import TradingAgentsGraph

            self._graph = TradingAgentsGraph(debug=self.debug, config=self._cfg)
        return self._graph

    def predict(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        graph = self._ensure_graph()
        # The framework's interface accepts a date string per its README.
        _, raw_decision = graph.propagate(ticker, as_of.isoformat())
        direction = self.decision_parser(raw_decision)
        conf = self.confidence_parser(raw_decision) if self.confidence_parser is not None else None
        if conf is not None and not (0.0 <= conf <= 1.0):
            conf = None  # silently drop out-of-range to avoid fabricating; calibration will report none-given
        return Decision(direction=direction, confidence=conf, raw={"tradingagents_raw": raw_decision})
