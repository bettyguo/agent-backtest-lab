"""WalkForwardEngine — drives a StrategyAdapter day-by-day through the leakage firewall.

Convention
----------
On trading day t, we call `adapter.predict(ticker, as_of=t, pit_data=PITView(as_of=t))`.
The returned Decision is the position to hold from t's close to (t+1)'s close, so the
return earned is `r_{t+1} = close_{t+1}/close_t - 1` for LONG, the negative for SHORT, 0 for FLAT.

This avoids the most common subtle look-ahead bug: a strategy that "decides" on day t
using day-t close information and is then credited with day-t's return. We never do that.
The decision on day t earns day-(t+1)'s return.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from abl.config import DEFAULT_COST_BPS
from abl.costs.models import ConstantBpsCost, CostModel
from abl.data.firewall import Firewall
from abl.data.loaders import DataStore
from abl.data.pit_view import PITView
from abl.types import BacktestResult, Decision, StrategyAdapter, Universe, Window


@dataclass
class EngineConfig:
    """Knobs for the engine. Sensible defaults; explicit is better than implicit."""

    cost_model: CostModel = field(default_factory=lambda: ConstantBpsCost(DEFAULT_COST_BPS))
    sizing: float = 1.0  # per-leg notional as fraction of equity (LONG: +sizing, SHORT: -sizing)
    equal_weight: bool = True  # split sizing across active tickers each day
    skip_missing: bool = True  # if a ticker has no bars by as_of, treat as FLAT and continue


def _direction_to_position(direction: str, sizing: float) -> float:
    if direction == "LONG":
        return +sizing
    if direction == "SHORT":
        return -sizing
    return 0.0


@dataclass
class WalkForwardEngine:
    """Executes the StrategyAdapter day-by-day over the universe.

    The engine is the only place that constructs PITView objects. Adapters never
    construct one themselves, so the firewall mediates 100% of adapter-driven access.
    Whether an adapter does its own out-of-band fetching (e.g. LLM tool use) is what
    the post-hoc leakage detectors are for.
    """

    store: DataStore
    config: EngineConfig = field(default_factory=EngineConfig)

    def run(
        self,
        adapter: StrategyAdapter,
        universe: Universe,
        window: Window,
    ) -> BacktestResult:
        # Trading calendar: union of bar dates for tickers in universe, clipped to window.
        cal = self.store.trading_calendar(universe.tickers)
        cal = cal[(cal >= pd.Timestamp(window.start)) & (cal <= pd.Timestamp(window.end))]
        if len(cal) < 2:
            raise ValueError(
                "Window contains fewer than 2 trading days; cannot compute next-day return."
            )

        firewall = Firewall(strict=True)
        decisions_rows: list[dict] = []

        # Maintain previous-day position per ticker (for turnover).
        prev_position: dict[str, float] = {t: 0.0 for t in universe.tickers}

        # Per-day net portfolio return series.
        portfolio_gross_returns: list[float] = []
        portfolio_net_returns: list[float] = []
        portfolio_turnover: list[float] = []
        portfolio_dates: list[pd.Timestamp] = []

        trades_rows: list[dict] = []

        n_active_today = max(1, len(universe.tickers)) if self.config.equal_weight else 1

        # Iterate up to the second-to-last day so we always have day-(t+1) to score.
        for i in range(len(cal) - 1):
            t = cal[i].date()
            t_next = cal[i + 1]
            pit = PITView(self.store, t, firewall)

            today_pos: dict[str, float] = {}
            today_returns: dict[str, float] = {}

            for ticker in universe.tickers:
                # Score next-day return based on raw close (close_{t+1}/close_t - 1).
                bars_t = self.store.raw_bars(ticker)
                # We can read bars_t freely here — this is the engine computing PnL, not the agent.
                if bars_t.empty or pd.Timestamp(t) not in bars_t.index or t_next not in bars_t.index:
                    if not self.config.skip_missing:
                        raise KeyError(f"Missing bars for {ticker} on {t} or {t_next}")
                    today_pos[ticker] = 0.0
                    today_returns[ticker] = 0.0
                    continue

                # Ask the adapter for its decision on day t.
                try:
                    decision = adapter.predict(ticker, t, pit)
                except Exception as exc:
                    # An adapter exception is recorded but does not crash the run; treat as FLAT.
                    decision = Decision(direction="FLAT", confidence=None, raw={"error": repr(exc)})

                pos = _direction_to_position(decision.direction, self.config.sizing) / n_active_today
                today_pos[ticker] = pos

                close_t = float(bars_t.loc[pd.Timestamp(t), "close"])
                close_t_next = float(bars_t.loc[t_next, "close"])
                ret_t_next = (close_t_next / close_t) - 1.0
                today_returns[ticker] = pos * ret_t_next

                decisions_rows.append(
                    {
                        "as_of": pd.Timestamp(t),
                        "ticker": ticker,
                        "direction": decision.direction,
                        "confidence": decision.confidence,
                        "position": pos,
                        "ret_next": ret_t_next,
                    }
                )

            # Aggregate to portfolio level.
            day_gross = float(sum(today_returns.values()))
            day_turnover = float(sum(abs(today_pos[t] - prev_position[t]) for t in universe.tickers))
            day_cost_series = self.config.cost_model.cost_fraction(
                pd.Series({pd.Timestamp(t): day_turnover})
            )
            day_cost = float(day_cost_series.iloc[0])
            day_net = day_gross - day_cost

            portfolio_dates.append(pd.Timestamp(t_next))
            portfolio_gross_returns.append(day_gross)
            portfolio_net_returns.append(day_net)
            portfolio_turnover.append(day_turnover)

            for ticker in universe.tickers:
                delta = today_pos[ticker] - prev_position[ticker]
                if delta != 0:
                    trades_rows.append(
                        {
                            "as_of": pd.Timestamp(t),
                            "ticker": ticker,
                            "direction_change": delta,
                            "notional": abs(delta),
                        }
                    )
                prev_position[ticker] = today_pos[ticker]

        return BacktestResult(
            adapter_name=adapter.name,
            universe=universe,
            window=window,
            decisions=pd.DataFrame(decisions_rows),
            daily_pnl_net=pd.Series(portfolio_net_returns, index=portfolio_dates, name="net_return"),
            daily_pnl_gross=pd.Series(portfolio_gross_returns, index=portfolio_dates, name="gross_return"),
            trades=pd.DataFrame(trades_rows),
            audit_events=[ev.__dict__ for ev in firewall.events],
            extras={"turnover": pd.Series(portfolio_turnover, index=portfolio_dates, name="turnover")},
        )
