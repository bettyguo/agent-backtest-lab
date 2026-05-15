"""Global defaults and the single-source-of-truth disclaimer strings.

DO NOT inline the disclaimer text elsewhere — import from here.
"""
from __future__ import annotations

DEFAULT_COST_BPS: float = 5.0  # each side; round-trip = 10 bps. Conservative-but-realistic.
DEFAULT_ANNUALIZATION: int = 252  # trading days per year for daily-bar US equities
DEFAULT_SEED: int = 20260514  # date of this release; used wherever stochasticity is reproducible
DEFAULT_CONFORMAL_ALPHA: float = 0.1  # 90% nominal coverage by default
DEFAULT_BH_ALPHA: float = 0.05  # standard BH-FDR control level
DEFAULT_PBO_BLOCKS: int = 16  # CSCV partitions; even, divisible into many (S/2)-combinations

DISCLAIMER_LINE: str = (
    "agent-backtest-lab is a research tool. Not financial advice. Not a trading system. "
    "Backtests don't predict the future."
)

DISCLAIMER_BLOCK: str = (
    "This project is a research and evaluation tool. It is not financial,\n"
    "investment, or trading advice. It does not execute trades or connect to\n"
    "brokerages. It exists to help researchers and practitioners rigorously\n"
    "measure how trading-agent frameworks actually perform — including, and\n"
    "especially, when they perform badly. Backtest results are not predictive\n"
    "of live performance. You are responsible for any decisions you make."
)

ATTRIBUTION: str = (
    "Built by Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong, "
    "advised by Prof. Siu-Ming Yiu. ORCID: 0009-0000-2388-1072. Apache-2.0."
)
