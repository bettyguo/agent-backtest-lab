"""Sortino ratio, Information Ratio, and downside-only risk metrics.

These complement the Sharpe ratio with metrics that distinguish good vs bad volatility
or compare to a benchmark rather than to cash.

Sources
-------
- Sortino, F. A., & Price, L. N. (1994). Performance Measurement in a Downside Risk
  Framework. *Journal of Investing*, 3(3), 59-64.
- Information Ratio is a standard quant metric; the canonical reference is
  Grinold & Kahn, *Active Portfolio Management*, McGraw-Hill, 2000 (2nd ed.).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abl.config import DEFAULT_ANNUALIZATION


@dataclass(frozen=True)
class RiskMetrics:
    sortino_annualized: float          # mean / downside_dev * sqrt(annualization)
    information_ratio_annualized: float  # (mean(r) − mean(bench)) / std(r − bench) * sqrt(annualization)
    downside_deviation: float          # sqrt(mean(min(r − target, 0)^2)) at observed cadence
    target_return: float               # MAR target used for Sortino


def sortino_ratio(
    returns: np.ndarray,
    *,
    target_return: float = 0.0,
    annualization: int = DEFAULT_ANNUALIZATION,
) -> float:
    """Sortino ratio at the input cadence, annualized.

    `target_return` is the Minimum Acceptable Return (MAR). 0 is the most common choice
    (cash); pass a daily benchmark return if you want a risk-free-rate adjustment.

    Downside deviation uses the second moment of `min(r - target, 0)` divided by N
    (NOT N-1; this is the conventional Sortino denominator — see Sortino & Price 1994).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - target_return
    downside = np.minimum(excess, 0.0)
    dd2 = float(np.mean(downside ** 2))
    if dd2 <= 0:
        return float("nan")
    dd = float(np.sqrt(dd2))
    return float(excess.mean() / dd * np.sqrt(annualization))


def information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    annualization: int = DEFAULT_ANNUALIZATION,
) -> float:
    """Annualized Information Ratio: tracking-error-normalized excess return vs benchmark.

    IR = mean(r - b) / std(r - b, ddof=1) * sqrt(annualization).
    """
    r = np.asarray(strategy_returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    n = min(r.size, b.size)
    r = r[:n]
    b = b[:n]
    mask = np.isfinite(r) & np.isfinite(b)
    r = r[mask]
    b = b[mask]
    if r.size < 2:
        return float("nan")
    active = r - b
    te = float(active.std(ddof=1))
    if te <= 0:
        return float("nan")
    return float(active.mean() / te * np.sqrt(annualization))


def risk_metrics_summary(
    returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    *,
    target_return: float = 0.0,
    annualization: int = DEFAULT_ANNUALIZATION,
) -> RiskMetrics:
    """Bundled Sortino + IR computation. IR is NaN if no benchmark is given."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return RiskMetrics(
            sortino_annualized=float("nan"),
            information_ratio_annualized=float("nan"),
            downside_deviation=float("nan"),
            target_return=target_return,
        )
    excess = r - target_return
    downside = np.minimum(excess, 0.0)
    dd2 = float(np.mean(downside ** 2))
    dd = float(np.sqrt(dd2)) if dd2 > 0 else 0.0
    sortino = sortino_ratio(returns, target_return=target_return, annualization=annualization)
    ir = (
        information_ratio(returns, benchmark_returns, annualization=annualization)
        if benchmark_returns is not None
        else float("nan")
    )
    return RiskMetrics(
        sortino_annualized=sortino,
        information_ratio_annualized=ir,
        downside_deviation=dd,
        target_return=target_return,
    )
