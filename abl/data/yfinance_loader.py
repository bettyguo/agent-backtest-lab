"""yfinance loader — RAW-ONLY (no retrospectively-adjusted Adj Close).

Why this exists
---------------
yfinance is the most common free data source. Its `Adj Close` column is RETROACTIVELY
adjusted for every split and dividend, INCLUDING ones that happened AFTER the bar's
date. A strategy backtested on `Adj Close` can earn returns from trading patterns that
only became visible after a future corporate action — the single most common silent
leakage in retail backtests.

This loader REFUSES to read `Adj Close`. It pulls raw OHLCV plus dividends/splits and
returns a DataStore whose `raw_bars()` is the unadjusted tape and whose
`corporate_actions()` is the actions table — so the leakage firewall and PITView can
reconstruct the adjusted series using only actions known at as_of.

Caveats
-------
- yfinance is an unofficial Yahoo scraper. Yahoo's ToS is the binding constraint.
- Symbol changes / delistings can silently lose data; survivorship-bias-free coverage
  is NOT provided. The scorecard's `SURVIVORSHIP_UNVERIFIED` flag will fire.
- yfinance versions have differed on the default `auto_adjust` behavior. We force
  `auto_adjust=False` and read raw OHLC explicitly.
- This loader is an OPTIONAL dependency. `pip install "agent-backtest-lab[yfinance]"`.

Usage
-----
    from abl.data.yfinance_loader import load_yfinance

    store = load_yfinance(
        tickers=("SPY", "AAPL", "MSFT"),
        start="2019-01-01",
        end="2024-12-31",
    )
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd

from abl.data.loaders import DataStore


def _require_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ImportError as e:
        raise ImportError(
            "yfinance is an optional dependency. Install with: "
            'pip install "agent-backtest-lab[yfinance]"'
        ) from e
    return yf


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Extract raw OHLCV from a yfinance DataFrame in a version-stable way.

    yfinance returns columns 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
    We DELIBERATELY drop 'Adj Close' and rename the rest. The leakage firewall + PITView
    + corporate_actions reconstruct adjusted prices at as_of using only known actions.
    """
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index.tz_localize(None) if df.index.tz is not None else df.index)
    out.index.name = "date"
    if "Open" in df.columns:
        out["open"] = df["Open"].astype(float)
    if "High" in df.columns:
        out["high"] = df["High"].astype(float)
    if "Low" in df.columns:
        out["low"] = df["Low"].astype(float)
    if "Close" in df.columns:
        out["close"] = df["Close"].astype(float)
    if "Volume" in df.columns:
        out["volume"] = df["Volume"].astype(float)
    # If multi-index columns (multi-ticker download with non-list usage), fail loudly.
    if any(isinstance(c, tuple) for c in df.columns):
        raise ValueError(
            "yfinance returned multi-index columns; pass tickers one at a time to this loader "
            "or use yf.Ticker(...).history()."
        )
    return out


def _extract_corporate_actions(
    dividends: pd.Series | None, splits: pd.Series | None
) -> pd.DataFrame:
    """Turn yfinance's dividend + split series into our corporate_actions schema."""
    rows: list[dict] = []
    if dividends is not None and not dividends.empty:
        for d_ts, amount in dividends.items():
            ts = pd.Timestamp(d_ts)
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            if float(amount) > 0:
                rows.append(
                    {"ex_date": ts.date(), "kind": "cash_div", "ratio": 1.0, "amount": float(amount)}
                )
    if splits is not None and not splits.empty:
        for d_ts, ratio in splits.items():
            ts = pd.Timestamp(d_ts)
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            if float(ratio) > 0 and float(ratio) != 1.0:
                rows.append(
                    {"ex_date": ts.date(), "kind": "split", "ratio": float(ratio), "amount": 0.0}
                )
    if not rows:
        return pd.DataFrame(
            {
                "ex_date": pd.Series(dtype="object"),
                "kind": pd.Series(dtype="object"),
                "ratio": pd.Series(dtype="float64"),
                "amount": pd.Series(dtype="float64"),
            }
        )
    return pd.DataFrame(rows).sort_values("ex_date").reset_index(drop=True)


def load_yfinance(
    tickers: Iterable[str],
    *,
    start: str | date,
    end: str | date,
) -> DataStore:
    """Load raw OHLCV + corporate actions from yfinance into a DataStore.

    See module docstring for the rationale — `Adj Close` is REFUSED.
    """
    yf = _require_yfinance()
    if isinstance(start, date):
        start = start.isoformat()
    if isinstance(end, date):
        end = end.isoformat()
    bars: dict[str, pd.DataFrame] = {}
    actions: dict[str, pd.DataFrame] = {}
    for t in tickers:
        ticker_obj = yf.Ticker(t)
        # `auto_adjust=False` is critical — we want RAW close, not back-adjusted.
        hist = ticker_obj.history(
            start=start, end=end, auto_adjust=False, actions=True, raise_errors=False
        )
        if hist is None or hist.empty:
            # Empty bars: caller will see no data and the engine will skip the ticker.
            bars[t] = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            actions[t] = _extract_corporate_actions(None, None)
            continue
        bars[t] = _normalize_bars(hist)
        # yfinance's `actions=True` populates the .dividends / .splits attributes too,
        # but the in-DataFrame 'Dividends' and 'Stock Splits' columns are the more
        # version-stable surface.
        div_series = hist["Dividends"] if "Dividends" in hist.columns else None
        split_series = hist["Stock Splits"] if "Stock Splits" in hist.columns else None
        # Filter to non-zero entries
        if div_series is not None:
            div_series = div_series[div_series != 0]
        if split_series is not None:
            split_series = split_series[split_series != 0]
        actions[t] = _extract_corporate_actions(div_series, split_series)
    return DataStore(_bars=bars, _actions=actions)
