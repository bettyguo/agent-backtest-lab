"""PITView — the as-of-restricted view of the data store.

Every adapter receives one of these. Every method routes through the firewall and
hard-refuses any access that would expose future data. The adjusted OHLCV returned
is reconstructed from raw bars + only-known-at-as_of corporate actions.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from abl.data.corporate_actions import apply_actions_as_of
from abl.data.firewall import Firewall

if TYPE_CHECKING:
    from abl.data.loaders import DataStore


class PITView:
    """An as-of-restricted, leakage-firewalled view of the data store."""

    __slots__ = ("_store", "_as_of", "_firewall")

    def __init__(self, store: DataStore, as_of: date, firewall: Firewall) -> None:
        self._store = store
        self._as_of = as_of
        self._firewall = firewall

    def as_of(self) -> date:
        return self._as_of

    def bars(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        """Return adjusted OHLCV ending at as_of (inclusive), at most `lookback_days` bars back.

        Adjustments are computed using only corporate actions with ex_date <= as_of, so the
        series the adapter sees is exactly what a real-time user would have seen on as_of.
        """
        if lookback_days < 1:
            raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
        requested_min = self._as_of - timedelta(days=lookback_days * 2)  # calendar slack
        self._firewall.check(
            as_of=self._as_of,
            ticker=ticker,
            kind="bars",
            requested_min_date=requested_min,
            requested_max_date=self._as_of,
        )
        raw = self._store.raw_bars(ticker)
        if raw is None or raw.empty:
            return raw.iloc[0:0] if raw is not None else pd.DataFrame()
        # Defensive: hard-clip to as_of regardless of what raw contains
        clipped = raw[raw.index <= pd.Timestamp(self._as_of)]
        if clipped.empty:
            return clipped
        # Apply only actions known at as_of
        actions = self._store.corporate_actions(ticker)
        adjusted = apply_actions_as_of(clipped, actions, self._as_of)
        # Take last `lookback_days` trading bars
        return adjusted.iloc[-lookback_days:].copy()

    def corporate_actions(self, ticker: str) -> pd.DataFrame:
        """Return corporate actions with ex_date <= as_of."""
        self._firewall.check(
            as_of=self._as_of,
            ticker=ticker,
            kind="corporate_actions",
            requested_min_date=date(1900, 1, 1),
            requested_max_date=self._as_of,
        )
        all_actions = self._store.corporate_actions(ticker)
        if all_actions.empty:
            return all_actions.copy()
        return all_actions[all_actions["ex_date"] <= self._as_of].copy()

    def universe_tickers(self) -> tuple[str, ...]:
        """Return the tickers known to the underlying store. Metadata access; no future-date risk."""
        self._firewall.check(
            as_of=self._as_of,
            ticker="_universe_",
            kind="metadata",
            requested_min_date=self._as_of,
            requested_max_date=self._as_of,
        )
        return tuple(self._store.tickers())
