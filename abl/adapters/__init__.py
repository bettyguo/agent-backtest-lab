"""Strategy adapters — the interface every framework must satisfy.

The protocol is in `abl.types.StrategyAdapter`. Three concrete adapters live here:

- `CallableAdapter` — wraps any callable. The escape hatch for custom frameworks.
- `PlainStrategyAdapter` — used internally by the built-in baselines.
- `TradingAgentsAdapter` — optional dependency on TauricResearch/TradingAgents.
"""
from __future__ import annotations

from abl.adapters.callable_adapter import CallableAdapter
from abl.adapters.fingpt import FinGPTAdapter
from abl.adapters.finrobot import FinRobotAdapter
from abl.adapters.plain_strategy import PlainStrategyAdapter
from abl.adapters.tradingagents import TradingAgentsAdapter

__all__ = [
    "CallableAdapter",
    "PlainStrategyAdapter",
    "TradingAgentsAdapter",
    "FinGPTAdapter",
    "FinRobotAdapter",
]
