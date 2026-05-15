"""Tests for the yfinance loader. Network-free — yfinance is mocked.

We never make a real Yahoo call in CI. The loader's job is to (a) refuse `Adj Close`
and (b) turn yfinance's return shape into our DataStore schema. Both are easily
verifiable without network access.
"""
from __future__ import annotations

import sys
import types
from datetime import date

import pandas as pd
import pytest

from abl.data.yfinance_loader import _extract_corporate_actions, _normalize_bars


def test_normalize_bars_drops_adj_close():
    """The loader MUST NOT carry 'Adj Close' through to our store schema.

    This is the load-bearing assertion: Yahoo's `Adj Close` is the retroactively-
    adjusted series that drives the most common silent leakage in retail backtests.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    src = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 99.0],
            "High": [102.0, 102.5, 100.5],
            "Low": [99.5, 100.0, 98.0],
            "Close": [101.0, 102.0, 99.5],
            "Adj Close": [50.5, 51.0, 49.75],  # back-adjusted via a 2:1 split later
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=idx,
    )
    out = _normalize_bars(src)
    assert "adj_close" not in out.columns
    assert "Adj Close" not in out.columns
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["close"].tolist() == [101.0, 102.0, 99.5]


def test_normalize_bars_handles_empty():
    df = pd.DataFrame()
    out = _normalize_bars(df)
    assert out.empty


def test_extract_corporate_actions_dividends():
    idx = pd.DatetimeIndex(pd.to_datetime(["2020-03-15", "2020-06-15"]))
    divs = pd.Series([0.25, 0.30], index=idx)
    out = _extract_corporate_actions(divs, None)
    assert len(out) == 2
    assert out.iloc[0]["kind"] == "cash_div"
    assert out.iloc[0]["amount"] == pytest.approx(0.25)
    assert out.iloc[0]["ex_date"] == date(2020, 3, 15)


def test_extract_corporate_actions_splits():
    idx = pd.DatetimeIndex(pd.to_datetime(["2020-08-31"]))
    splits = pd.Series([4.0], index=idx)
    out = _extract_corporate_actions(None, splits)
    assert len(out) == 1
    assert out.iloc[0]["kind"] == "split"
    assert out.iloc[0]["ratio"] == pytest.approx(4.0)


def test_extract_corporate_actions_filters_zeros():
    idx = pd.DatetimeIndex(pd.to_datetime(["2020-01-01", "2020-04-01"]))
    divs = pd.Series([0.0, 0.25], index=idx)
    out = _extract_corporate_actions(divs, None)
    assert len(out) == 1
    assert out.iloc[0]["ex_date"] == date(2020, 4, 1)


def test_yfinance_import_error_when_missing(monkeypatch):
    """When yfinance is not installed, the loader must raise a clean ImportError."""
    # Force the import lookup to fail
    monkeypatch.setitem(sys.modules, "yfinance", None)
    from abl.data.yfinance_loader import _require_yfinance
    with pytest.raises(ImportError) as exc_info:
        _require_yfinance()
    assert "yfinance" in str(exc_info.value)


def test_load_yfinance_with_stub(monkeypatch):
    """End-to-end load via a stub yfinance module."""
    # Build a fake yfinance namespace
    class _FakeTicker:
        def __init__(self, sym):
            self.sym = sym

        def history(self, **kwargs):
            assert kwargs.get("auto_adjust") is False  # REQUIRED — we never accept auto-adjust
            idx = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-03"]))
            return pd.DataFrame(
                {
                    "Open": [100.0, 101.0],
                    "High": [102.0, 102.5],
                    "Low": [99.5, 100.0],
                    "Close": [101.0, 102.0],
                    "Adj Close": [50.5, 51.0],
                    "Volume": [1_000_000, 1_100_000],
                    "Dividends": [0.0, 0.25],
                    "Stock Splits": [0.0, 0.0],
                },
                index=idx,
            )

    fake = types.SimpleNamespace(Ticker=_FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    from abl.data.yfinance_loader import load_yfinance
    store = load_yfinance(("AAPL",), start="2020-01-01", end="2020-01-31")
    bars = store.raw_bars("AAPL")
    assert "adj_close" not in bars.columns
    actions = store.corporate_actions("AAPL")
    assert len(actions) == 1
    assert actions.iloc[0]["kind"] == "cash_div"
