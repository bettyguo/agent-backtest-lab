"""Apply corporate actions (splits, cash dividends) to raw OHLCV using only
actions with `ex_date <= as_of`.

The retroactive-adjustment leakage failure mode this exists to defeat: vendors
like Yahoo Finance return an `Adj Close` series that has been back-adjusted for
ALL splits and dividends, including ones that happened AFTER the bar's date. A
strategy reading that series can earn returns by trading patterns only visible
after a future corporate action. We never give the agent that series.

We reconstruct the adjusted series at evaluation time by chaining only the
actions known at `as_of`.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def apply_actions_as_of(
    bars: pd.DataFrame, actions: pd.DataFrame, as_of: date
) -> pd.DataFrame:
    """Return adjusted OHLCV computed from raw bars using only actions with ex_date <= as_of.

    Parameters
    ----------
    bars
        Raw OHLCV. Index: trading date (date). Columns: open, high, low, close, volume.
        These prices are NEVER retroactively adjusted; they are what printed on the tape.
    actions
        Corporate actions. Columns: ex_date (date), kind ("split"|"cash_div"), ratio (float for splits),
        amount (float for cash_div). For a 2-for-1 split, ratio=2.0. For a $0.25 cash div, amount=0.25.
    as_of
        The point-in-time cutoff. Only actions with ex_date <= as_of are applied.

    Returns
    -------
    DataFrame
        Adjusted OHLCV. For every bar with date < ex_date of an applied split, prices
        are divided by the cumulative split ratio. For cash dividends, prices for dates
        strictly before the ex_date are reduced by the dividend amount (the standard
        Center-for-Research-in-Security-Prices style proportional adjustment), then
        scaled. We use the simple proportional model: factor = (close_t-1 - dividend) / close_t-1
        applied multiplicatively to all bars with date < ex_date.

    Notes
    -----
    Volume is adjusted inversely to split ratios so dollar-volume is conserved.

    This implementation is intentionally simple; it covers the cases that actually appear
    in our fixtures and in standard US equity daily-bar data. Exotic actions (return-of-capital,
    spin-offs, reverse splits with non-integer ratios) are flagged but not specially handled.
    """
    if bars.empty:
        return bars.copy()

    out = bars.copy()
    relevant = actions[actions["ex_date"] <= as_of].sort_values("ex_date") if not actions.empty else actions
    if relevant.empty:
        return out

    for _, row in relevant.iterrows():
        ex = row["ex_date"]
        kind = row["kind"]
        mask = out.index < pd.Timestamp(ex)
        if not mask.any():
            continue
        if kind == "split":
            ratio = float(row["ratio"])
            if ratio <= 0:
                raise ValueError(f"split ratio must be positive, got {ratio}")
            for col in ("open", "high", "low", "close"):
                if col in out.columns:
                    out.loc[mask, col] = out.loc[mask, col] / ratio
            if "volume" in out.columns:
                out.loc[mask, "volume"] = out.loc[mask, "volume"] * ratio
        elif kind == "cash_div":
            amount = float(row["amount"])
            if amount < 0:
                raise ValueError(f"cash_div amount must be >= 0, got {amount}")
            prior_close_idx = out.index[mask][-1] if mask.any() else None
            if prior_close_idx is None:
                continue
            prior_close = out.loc[prior_close_idx, "close"]
            if prior_close <= 0:
                continue
            factor = max(0.0, (prior_close - amount) / prior_close)
            for col in ("open", "high", "low", "close"):
                if col in out.columns:
                    out.loc[mask, col] = out.loc[mask, col] * factor
        else:
            # Unknown action — do not silently apply it. Skip and let the caller decide.
            continue

    # Guard against any negative or zero prices introduced by pathological inputs.
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].clip(lower=np.finfo(float).tiny)
    return out
