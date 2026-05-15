"""Transaction-cost models.

The single rule of this module: returns reported to the user must be NET of costs.
We expose a small set of well-understood models and reject any pretense of modeling
microstructure that a daily-bar academic harness cannot honestly support.

References
----------
- Almgren & Chriss, "Optimal Execution of Portfolio Transactions," J. Risk, 2000.
  Linear permanent + linear temporary impact framework.
- Almgren, Thum, Hauptmann, Li, "Direct Estimation of Equity Market Impact," Risk
  (July 2005). Empirical fit on Citigroup US-equity trading data found temporary
  impact exponent close to 3/5 (NOT 1/2). The "square-root law" of impact remains
  the practitioner default because 0.5 is close to 0.6 and easier to reason about;
  we expose `beta` as a parameter so users can use either, but our docstring is
  explicit about which paper supports which.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from abl.config import DEFAULT_COST_BPS


class CostModel(Protocol):
    """Apply costs to a per-day notional-turnover series, returning per-day cost as a fraction of equity."""

    name: str

    def cost_fraction(self, turnover_fraction: pd.Series, context: dict | None = None) -> pd.Series:
        """Per-day cost as a fraction of equity. Subtract this from gross return to get net return."""
        ...


@dataclass
class ConstantBpsCost:
    """Constant per-trade cost in basis points of notional traded.

    `bps_each_side` is the cost charged on each side of a trade (buy or sell). A round-trip
    incurs `2 * bps_each_side`. Default 5 bps each side is conservative-but-realistic for
    retail-tier daily US-equity strategies.
    """

    bps_each_side: float = DEFAULT_COST_BPS
    name: str = "constant_bps"

    def cost_fraction(self, turnover_fraction: pd.Series, context: dict | None = None) -> pd.Series:
        # turnover_fraction is the absolute change in net exposure as fraction of equity.
        # E.g. flipping FLAT->LONG with full sizing is 1.0 of turnover; LONG->SHORT is 2.0.
        cost = turnover_fraction.abs() * (self.bps_each_side / 1e4)
        cost.name = "cost_fraction"
        return cost


@dataclass
class AlmgrenImpactCost:
    """Almgren-style temporary-impact cost: Δp/p ≈ σ · η · (Q / ADV)^β.

    Parameters
    ----------
    eta : float
        Impact coefficient. Order of magnitude 1.0 in the original paper; users
        should calibrate to their universe.
    beta : float
        Exponent. 0.5 (square-root law) is the practitioner default. 0.6 / 3/5
        is the Almgren-Thum-Hauptmann-Li (2005) empirical estimate.
    sigma_provider : callable | None
        Function (ticker, as_of) -> daily volatility estimate. If None, sigma=0.01
        is used (1%/day, generic large-cap order of magnitude).
    """

    eta: float = 1.0
    beta: float = 0.5
    sigma_default: float = 0.01
    name: str = "almgren_impact"

    def cost_fraction(self, turnover_fraction: pd.Series, context: dict | None = None) -> pd.Series:
        # context is expected to carry `participation` (Q/ADV) per day if available;
        # if not, we assume a small uniform participation rate from the turnover itself.
        # This is a conservative simplification — see docstring.
        participation = (
            context.get("participation") if context is not None and "participation" in context
            else turnover_fraction.abs().clip(lower=0.0, upper=1.0)
        )
        if isinstance(participation, pd.Series):
            part = participation.reindex(turnover_fraction.index).fillna(0.0)
        else:
            part = pd.Series(participation, index=turnover_fraction.index)
        impact = self.sigma_default * self.eta * np.power(part.clip(lower=0.0), self.beta)
        cost = turnover_fraction.abs() * impact
        cost.name = "cost_fraction"
        return cost


@dataclass
class CompositeCost:
    """Sum of multiple cost models. The standard scorecard uses ConstantBpsCost only by default."""

    models: tuple[CostModel, ...]
    name: str = "composite"

    def cost_fraction(self, turnover_fraction: pd.Series, context: dict | None = None) -> pd.Series:
        total = pd.Series(0.0, index=turnover_fraction.index)
        for m in self.models:
            total = total.add(m.cost_fraction(turnover_fraction, context), fill_value=0.0)
        total.name = "cost_fraction"
        return total
