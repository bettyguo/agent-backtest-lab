"""Reward-hacking / overfitting-to-the-window detection.

Distinct from CSCV/PBO: PBO answers "across N candidate strategies, is the in-sample-best
reliably bad OOS?" Reward hacking asks "for THIS one strategy, does it perform much
better on its training window than on holdout?" — even when no hyperparameter scan was
run, a strategy can be reward-hacked simply because its prompts / examples / memory
were tuned to the visible window.

Signals
-------

A. **IS/OOS Sharpe ratio drop.** Split the strategy's return series chronologically at
   `is_fraction` (default 0.7). If the IS Sharpe is meaningfully positive but the OOS
   Sharpe is materially worse, flag.

B. **Drawdown structure change.** If the max-drawdown OOS is much worse than the max-
   drawdown IS, flag.

C. **Confidence-correctness divergence.** When the strategy emits confidences: is the
   IS calibration (ECE) much better than the OOS calibration? A reward-hacked strategy
   often has memorized "high confidence" on IS examples and is poorly calibrated OOS.

These are heuristics, NOT statistical guarantees. The scorecard renders them as warn-
or critical-severity flags. PBO (cross-strategy) and reward-hacking (single-strategy)
are complementary diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from abl.calibration.reliability import expected_calibration_error
from abl.scorecard.drawdown import drawdown_stats


@dataclass(frozen=True)
class RewardHackingFlag:
    code: str
    severity: str  # "info" | "warn" | "critical"
    message: str
    is_value: float | None = None
    oos_value: float | None = None


def _sharpe_ann(r: np.ndarray, annualization: int) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(annualization))


def detect_reward_hacking(
    *,
    returns: pd.Series,
    decisions: pd.DataFrame | None = None,
    is_fraction: float = 0.7,
    annualization: int = 252,
) -> list[RewardHackingFlag]:
    """Run reward-hacking heuristics on a single strategy's BacktestResult.

    Parameters
    ----------
    returns : pd.Series
        Net-of-cost daily return series.
    decisions : pd.DataFrame or None
        If present, used for the calibration-divergence signal.
    is_fraction : float
        Fraction of the series to treat as in-sample. Must be in (0.3, 0.9).
    annualization : int
    """
    if not (0.3 < is_fraction < 0.9):
        raise ValueError("is_fraction must be in (0.3, 0.9)")
    r = returns.dropna().to_numpy()
    if r.size < 60:  # too short to split meaningfully
        return []
    cut = int(len(r) * is_fraction)
    if cut < 30 or len(r) - cut < 30:
        return []
    is_r, oos_r = r[:cut], r[cut:]
    flags: list[RewardHackingFlag] = []

    # Signal A: Sharpe drop
    sr_is = _sharpe_ann(is_r, annualization)
    sr_oos = _sharpe_ann(oos_r, annualization)
    if np.isfinite(sr_is) and np.isfinite(sr_oos):
        drop = sr_is - sr_oos
        if sr_is > 1.0 and drop > 1.5 and sr_oos < 0.5:
            flags.append(
                RewardHackingFlag(
                    code="SHARPE_DROP_IS_OOS",
                    severity="critical",
                    message=(
                        f"Sharpe collapses from IS={sr_is:.2f} (first {int(is_fraction*100)}% of window) "
                        f"to OOS={sr_oos:.2f}. Drop of {drop:.2f} is consistent with reward hacking "
                        "or window-specific overfitting."
                    ),
                    is_value=sr_is,
                    oos_value=sr_oos,
                )
            )
        elif sr_is > 0.5 and drop > 1.0:
            flags.append(
                RewardHackingFlag(
                    code="SHARPE_DROP_IS_OOS",
                    severity="warn",
                    message=(
                        f"Sharpe drops from IS={sr_is:.2f} to OOS={sr_oos:.2f}. "
                        "Worth investigating whether the strategy was tuned to the IS window."
                    ),
                    is_value=sr_is,
                    oos_value=sr_oos,
                )
            )

    # Signal B: Drawdown widening
    dd_is = drawdown_stats(is_r, annualization=annualization)
    dd_oos = drawdown_stats(oos_r, annualization=annualization)
    if np.isfinite(dd_is.max_drawdown) and np.isfinite(dd_oos.max_drawdown):
        if dd_oos.max_drawdown < -0.10 and dd_oos.max_drawdown < 1.5 * dd_is.max_drawdown - 0.05:
            flags.append(
                RewardHackingFlag(
                    code="DRAWDOWN_WIDENS_OOS",
                    severity="warn",
                    message=(
                        f"Max drawdown widens from IS={dd_is.max_drawdown:+.2%} to "
                        f"OOS={dd_oos.max_drawdown:+.2%}. Risk profile is materially worse OOS."
                    ),
                    is_value=float(dd_is.max_drawdown),
                    oos_value=float(dd_oos.max_drawdown),
                )
            )

    # Signal C: Calibration divergence (only if confidence is emitted)
    if decisions is not None and not decisions.empty and "confidence" in decisions.columns:
        dec = decisions.copy()
        # Sort decisions chronologically by as_of so the IS/OOS split aligns with `cut`.
        dec = dec.sort_values("as_of").reset_index(drop=True)
        has_conf = ~dec["confidence"].isna()
        if has_conf.sum() >= 100:
            non_flat = dec[dec["direction"] != "FLAT"]
            if len(non_flat) >= 100:
                non_flat = non_flat.reset_index(drop=True)
                conf = non_flat["confidence"].astype(float).to_numpy()
                sign_call = np.where(non_flat["direction"] == "LONG", 1, -1)
                ret_next = non_flat["ret_next"].astype(float).to_numpy()
                correct = ((sign_call == np.sign(ret_next)) & (ret_next != 0)).astype(int)
                mid = len(non_flat) // 2
                if mid >= 30 and len(non_flat) - mid >= 30:
                    ece_is = expected_calibration_error(conf[:mid], correct[:mid], n_bins=8)
                    ece_oos = expected_calibration_error(conf[mid:], correct[mid:], n_bins=8)
                    if np.isfinite(ece_is) and np.isfinite(ece_oos):
                        if ece_oos > 0.15 and ece_oos > ece_is + 0.10:
                            flags.append(
                                RewardHackingFlag(
                                    code="CALIBRATION_DIVERGES_OOS",
                                    severity="warn",
                                    message=(
                                        f"Expected Calibration Error rises from IS={ece_is:.3f} to "
                                        f"OOS={ece_oos:.3f}. Confidence is uncalibrated on holdout."
                                    ),
                                    is_value=float(ece_is),
                                    oos_value=float(ece_oos),
                                )
                            )

    return flags
