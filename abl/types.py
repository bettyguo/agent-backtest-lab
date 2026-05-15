"""Core dataclasses and protocols for agent-backtest-lab."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd

Direction = Literal["LONG", "SHORT", "FLAT"]


@dataclass(frozen=True)
class Decision:
    """A single point-in-time directional call from a strategy adapter.

    Attributes
    ----------
    direction : {"LONG", "SHORT", "FLAT"}
        The directional call for the next holding period.
    confidence : float | None
        Calibrated probability in [0, 1] if the framework emits one; None otherwise.
        We deliberately do NOT fabricate a confidence — None propagates through calibration
        diagnostics as "not evaluable, confidence not provided." This is honest.
    raw : dict | None
        Opaque framework-native output, retained for forensics.
    """

    direction: Direction
    confidence: float | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("LONG", "SHORT", "FLAT"):
            raise ValueError(f"direction must be LONG/SHORT/FLAT, got {self.direction!r}")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1] or None, got {self.confidence!r}")


@dataclass(frozen=True)
class Window:
    """Inclusive date window for evaluation. Both ends are explicit; no defaults."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start {self.start} > end {self.end}")


@dataclass(frozen=True)
class Universe:
    """Declared universe of tickers under evaluation.

    Required as a first-class API object so users cannot silently cherry-pick: the
    universe is a property of the *run*, not of the strategy.

    `survivorship_verified=False` (the default for free-data fixtures) triggers a
    `universe_flag` in the scorecard, since most public sources are biased.
    """

    tickers: tuple[str, ...]
    name: str
    survivorship_verified: bool = False
    source_note: str = ""

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("Universe.tickers must be non-empty")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("Universe.tickers must not contain duplicates")


@dataclass
class BacktestResult:
    """The output of WalkForwardEngine.run().

    Holds per-day decisions, per-day net returns (after cost), and the firewall
    audit log. The scorecard consumes this directly.
    """

    adapter_name: str
    universe: Universe
    window: Window
    decisions: pd.DataFrame  # cols: as_of, ticker, direction, confidence
    daily_pnl_net: pd.Series  # index: trading day, value: portfolio-level net return
    daily_pnl_gross: pd.Series  # internal; never rendered to user-facing reports
    trades: pd.DataFrame  # cols: as_of, ticker, direction_change, notional
    audit_events: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StrategyAdapter(Protocol):
    """Minimal interface every framework must satisfy to be evaluated.

    `pit_data` is always produced by the leakage firewall and refuses future data.
    Adapters that bypass it (e.g. by making their own network calls inside an LLM)
    cannot be hard-prevented from doing so by Python; the post-hoc leakage detectors
    exist precisely for that case. See `abl.leakage`.
    """

    name: str

    def predict(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision: ...


@runtime_checkable
class PITViewProtocol(Protocol):
    """Protocol satisfied by abl.data.pit_view.PITView. Kept here to avoid an import cycle."""

    def bars(self, ticker: str, lookback_days: int) -> pd.DataFrame: ...
    def corporate_actions(self, ticker: str) -> pd.DataFrame: ...
    def as_of(self) -> date: ...
