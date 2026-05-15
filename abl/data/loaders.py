"""Data store and loaders.

The DataStore is a tiny in-memory wrapper around the raw bars + corporate actions for
the universe under evaluation. The leakage firewall never reads the store directly; it
goes through the PITView which filters by as_of.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from abl.data.fixture_synth import SyntheticFixture, generate_synthetic_fixture


@dataclass
class DataStore:
    """In-memory raw OHLCV + corporate-actions store.

    `raw_bars(ticker)` returns the raw (unadjusted) OHLCV. The retroactive-adjustment
    leakage failure mode (Yahoo's `Adj Close` etc.) cannot happen here: we never store
    a pre-adjusted series.
    """

    _bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    _actions: dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def from_synthetic(cls, fix: SyntheticFixture) -> DataStore:
        return cls(_bars=dict(fix.bars), _actions=dict(fix.actions))

    def tickers(self) -> tuple[str, ...]:
        return tuple(self._bars.keys())

    def raw_bars(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._bars:
            return pd.DataFrame()
        return self._bars[ticker]

    def corporate_actions(self, ticker: str) -> pd.DataFrame:
        empty = pd.DataFrame(
            {"ex_date": pd.Series(dtype="object"), "kind": pd.Series(dtype="object"),
             "ratio": pd.Series(dtype="float64"), "amount": pd.Series(dtype="float64")}
        )
        return self._actions.get(ticker, empty)

    def add_corporate_action(
        self, ticker: str, *, ex_date, kind: str, ratio: float = 1.0, amount: float = 0.0
    ) -> None:
        existing = self._actions.get(ticker)
        new_row = pd.DataFrame(
            [{"ex_date": ex_date, "kind": kind, "ratio": ratio, "amount": amount}]
        )
        if existing is None or existing.empty:
            self._actions[ticker] = new_row
        else:
            self._actions[ticker] = pd.concat([existing, new_row], ignore_index=True)

    def trading_calendar(self, tickers: Iterable[str] | None = None) -> pd.DatetimeIndex:
        """Union of trading dates across the listed tickers (or all if None)."""
        chosen = list(tickers) if tickers is not None else list(self._bars.keys())
        all_idx = pd.DatetimeIndex([])
        for t in chosen:
            if t in self._bars and not self._bars[t].empty:
                all_idx = all_idx.union(self._bars[t].index)
        return all_idx.sort_values()


def load_fixture(seed: int | None = None) -> DataStore:
    """Load the standard synthetic fixture for quickstart / CI / examples."""
    if seed is None:
        return DataStore.from_synthetic(generate_synthetic_fixture())
    return DataStore.from_synthetic(generate_synthetic_fixture(seed=seed))
