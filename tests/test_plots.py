"""Tests for the matplotlib plot renderers. Headless (Agg)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from abl.plots.renderers import (
    plot_drawdown,
    plot_equity_curve,
    plot_reliability_diagram,
)


def test_plot_equity_curve_writes_png(tmp_path):
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    primary = pd.Series(rng.normal(0.0005, 0.01, size=100), index=idx)
    bah = pd.Series(rng.normal(0.0004, 0.01, size=100), index=idx)
    out = tmp_path / "equity.png"
    plot_equity_curve(primary, {"buy_and_hold": bah}, out, primary_name="my_strategy")
    assert out.exists()
    assert out.stat().st_size > 200


def test_plot_drawdown_writes_png(tmp_path):
    idx = pd.date_range("2020-01-01", periods=250, freq="B")
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.02, size=250)
    series = pd.Series(r, index=idx)
    out = tmp_path / "dd.png"
    plot_drawdown(series, out)
    assert out.exists()
    assert out.stat().st_size > 200


def test_plot_reliability_diagram_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, size=1000)
    labels = (rng.uniform(0, 1, size=1000) < probs).astype(int)
    out = tmp_path / "rel.png"
    plot_reliability_diagram(probs, labels, out, n_bins=10)
    assert out.exists()
    assert out.stat().st_size > 200


def test_plot_reliability_diagram_handles_empty_input(tmp_path):
    out = tmp_path / "rel_empty.png"
    # An array with one value in each of two bins; should still render.
    probs = np.array([0.05, 0.95])
    labels = np.array([0, 1])
    plot_reliability_diagram(probs, labels, out)
    assert out.exists()
