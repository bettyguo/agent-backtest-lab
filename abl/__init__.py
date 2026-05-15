"""agent-backtest-lab — statistical-audit harness for trading-agent frameworks.

Authored by Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong,
advised by Prof. Siu-Ming Yiu. ORCID: 0009-0000-2388-1072. Apache-2.0.

This is a research and evaluation tool. It is not financial, investment, or trading advice.
It does not execute trades or connect to brokerages. Backtest results are not predictive of
live performance.
"""
from __future__ import annotations

__version__ = "0.3.0"
__author__ = "Betty Guo (Dongxin Guo / 郭东欣)"
__orcid__ = "0009-0000-2388-1072"
__license__ = "Apache-2.0"

from abl.config import (
    DEFAULT_ANNUALIZATION,
    DEFAULT_COST_BPS,
    DEFAULT_SEED,
    DISCLAIMER_BLOCK,
    DISCLAIMER_LINE,
)
from abl.types import (
    BacktestResult,
    Decision,
    Direction,
    Universe,
    Window,
)

__all__ = [
    "__version__",
    "__author__",
    "__orcid__",
    "__license__",
    "Decision",
    "Direction",
    "Universe",
    "Window",
    "BacktestResult",
    "DISCLAIMER_LINE",
    "DISCLAIMER_BLOCK",
    "DEFAULT_COST_BPS",
    "DEFAULT_ANNUALIZATION",
    "DEFAULT_SEED",
]
