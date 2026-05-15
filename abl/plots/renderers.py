"""Matplotlib plot renderers (headless / Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from abl.calibration.reliability import reliability_diagram_data  # noqa: E402
from abl.config import DISCLAIMER_LINE  # noqa: E402


def _save(fig: plt.Figure, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def _footer(fig: plt.Figure) -> None:
    """Stamp the disclaimer at the bottom of every plot. Not removable."""
    fig.text(
        0.5, 0.01, DISCLAIMER_LINE,
        ha="center", va="bottom", fontsize=7, color="#666666", style="italic",
    )


def plot_equity_curve(
    primary_returns: pd.Series,
    baseline_returns: dict[str, pd.Series],
    out_path: str | Path,
    *,
    title: str = "Net equity (cost-adjusted)",
    primary_name: str = "strategy",
) -> Path:
    """Plot primary strategy equity vs each baseline. Net of cost only."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    primary_eq = (1.0 + primary_returns.fillna(0)).cumprod()
    ax.plot(primary_eq.index, primary_eq.values, label=primary_name, linewidth=2.0, color="#1f4e8c")
    for name, r in baseline_returns.items():
        eq = (1.0 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=name, linewidth=1.0, alpha=0.7)
    ax.axhline(1.0, color="#aaaaaa", linewidth=0.5, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("equity (starting at 1.0)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig)
    return _save(fig, out_path)


def plot_drawdown(
    primary_returns: pd.Series,
    out_path: str | Path,
    *,
    title: str = "Drawdown (net of cost)",
) -> Path:
    """Plot the drawdown curve with the max-DD trough annotated."""
    r = primary_returns.fillna(0)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    md_idx = int(np.argmin(dd.values))
    md = float(dd.values[md_idx])

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.35)
    ax.plot(dd.index, dd.values, color="#c0392b", linewidth=1.0)
    ax.axhline(0, color="#888888", linewidth=0.5)
    if len(dd) > md_idx >= 0:
        ax.annotate(
            f"max DD: {md:.2%}",
            xy=(dd.index[md_idx], md),
            xytext=(20, -10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#444"),
            fontsize=9,
            color="#222",
        )
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig)
    return _save(fig, out_path)


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    *,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> Path:
    """Bar plot of empirical accuracy per confidence bin, with the diagonal reference."""
    bins = reliability_diagram_data(np.asarray(probs), np.asarray(labels), n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 6))
    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.0, label="perfect calibration")
    if bins:
        centers = [(b.lo + b.hi) / 2.0 for b in bins]
        accs = [b.empirical_accuracy for b in bins]
        widths = [(b.hi - b.lo) * 0.9 for b in bins]
        confs = [b.mean_confidence for b in bins]
        ax.bar(centers, accs, width=widths, color="#1f4e8c", alpha=0.6, edgecolor="black",
               label="empirical accuracy")
        ax.scatter(confs, accs, color="#c0392b", s=30, zorder=5, label="mean confidence")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig)
    return _save(fig, out_path)
