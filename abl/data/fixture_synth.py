"""Synthetic GBM fixture generator.

This is the primary CI fixture and the offline-quickstart dataset. We use synthetic
data deliberately:

1. We KNOW the true generating parameters (drift, vol, true directional probability).
   This lets every statistical fixture test verify against a ground-truth answer.
2. We avoid bundling vendor data with unclear redistribution terms.
3. We can inject specific pathologies (a leakage backdoor, an overfittable signal)
   that the gate tests rely on.

The generator is seeded; same seed gives bitwise-identical output across machines.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from abl.config import DEFAULT_SEED


@dataclass
class SyntheticFixture:
    """A small bundle of synthetic OHLCV time series with known generating parameters."""

    bars: dict[str, pd.DataFrame]  # ticker -> raw OHLCV
    actions: dict[str, pd.DataFrame]  # ticker -> corporate actions (typically empty)
    truth: dict[str, dict[str, float]]  # ticker -> {"mu": float, "sigma": float}

    def tickers(self) -> tuple[str, ...]:
        return tuple(self.bars.keys())


def generate_synthetic_fixture(
    *,
    tickers: tuple[str, ...] = ("SYN-A", "SYN-B", "SYN-C"),
    start: date = date(2020, 1, 6),  # Monday
    n_days: int = 1260,  # ~5 trading years
    mu_annual: float | tuple[float, ...] = 0.05,
    sigma_annual: float | tuple[float, ...] = 0.20,
    init_price: float = 100.0,
    seed: int = DEFAULT_SEED,
) -> SyntheticFixture:
    """Generate a deterministic GBM fixture with weekday-only ('trading') dates.

    The drift and vol can be a scalar (same for all tickers) or per-ticker tuples.
    Daily mean log-return = mu_annual / 252; daily log-return vol = sigma_annual / sqrt(252).

    No corporate actions in the default fixture. Tests that exercise the
    leakage-firewall corporate-action path inject their own actions.
    """
    rng = np.random.default_rng(seed)

    if isinstance(mu_annual, (int, float)):
        mus = (float(mu_annual),) * len(tickers)
    else:
        mus = mu_annual
    if isinstance(sigma_annual, (int, float)):
        sigmas = (float(sigma_annual),) * len(tickers)
    else:
        sigmas = sigma_annual
    if len(mus) != len(tickers) or len(sigmas) != len(tickers):
        raise ValueError("mu_annual / sigma_annual length mismatch with tickers")

    # Trading-day calendar: weekdays only. Good enough for fixture purposes.
    dates: list[date] = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d = d + timedelta(days=1)
    idx = pd.DatetimeIndex([pd.Timestamp(x) for x in dates])

    bars: dict[str, pd.DataFrame] = {}
    actions: dict[str, pd.DataFrame] = {}
    truth: dict[str, dict[str, float]] = {}

    for ticker, mu_a, sigma_a in zip(tickers, mus, sigmas, strict=True):
        mu_d = mu_a / 252.0
        sigma_d = sigma_a / np.sqrt(252.0)
        eps = rng.standard_normal(n_days)
        log_rets = mu_d - 0.5 * sigma_d**2 + sigma_d * eps
        close = init_price * np.exp(np.cumsum(log_rets))
        # OHLC: build plausible intraday range as fraction of close.
        intraday_range = np.abs(rng.standard_normal(n_days)) * sigma_d * close
        high = close + intraday_range / 2
        low = np.maximum(close - intraday_range / 2, 0.01)
        prev_close = np.concatenate([[init_price], close[:-1]])
        open_ = (prev_close + close) / 2  # naive but adequate for a fixture
        volume = (1_000_000 + rng.standard_normal(n_days) * 100_000).clip(min=10_000).astype(int)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
        df.index.name = "date"
        bars[ticker] = df
        actions[ticker] = pd.DataFrame(
            {"ex_date": pd.Series(dtype="object"), "kind": pd.Series(dtype="object"),
             "ratio": pd.Series(dtype="float64"), "amount": pd.Series(dtype="float64")}
        )
        truth[ticker] = {"mu_annual": float(mu_a), "sigma_annual": float(sigma_a)}

    return SyntheticFixture(bars=bars, actions=actions, truth=truth)
