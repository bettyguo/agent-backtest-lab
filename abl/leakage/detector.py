"""Post-hoc leakage detectors.

Each detector returns a list of LeakageFlag. The scorecard renders any non-empty flag
list prominently and turns the report banner red.

Heuristics
----------

A. **Firewall-violation flag.** If the firewall recorded any denied access, we flag it.
   In product code this should always be zero (the firewall raises on violation). If
   `strict=False` was used (a fault-injection test), violations appear here.

B. **Impossible-accuracy flag.** Per-day directional accuracy above a threshold over
   the whole window is a strong signal of leakage. We compute hit rate on tickers
   for which the agent issued a directional call, and flag if the binomial test p-value
   against "agent has no edge" (p=0.5 on LONG vs not-LONG, conditional on the call
   being non-FLAT) is below a strict threshold AND the absolute hit rate is implausibly
   high (default > 0.85). This is intentionally conservative — we'd rather miss a
   marginal leak than false-positive on a genuinely-good strategy.

C. **Corporate-action proximity flag.** Count correct directional calls in the window
   `[-2, 0]` trading days around any split/dividend ex-date. If the conditional hit
   rate is significantly above the unconditional baseline rate (one-sided z-test,
   p < 0.01), flag.

These are heuristics. They are documented as such in the scorecard. False positives
are possible; users are instructed to investigate flagged runs by hand. False negatives
are also possible — the hard firewall is the only thing that GUARANTEES no leakage
through adapter-mediated channels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

Severity = Literal["info", "warn", "critical"]


@dataclass(frozen=True)
class LeakageFlag:
    code: str
    severity: Severity
    message: str
    statistic: float | None = None
    pvalue: float | None = None


@dataclass
class LeakageReport:
    flags: list[LeakageFlag]

    @property
    def any_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.flags)

    @property
    def any(self) -> bool:
        return len(self.flags) > 0


def _firewall_violation_flag(audit_events: list[dict]) -> list[LeakageFlag]:
    violations = [e for e in audit_events if not e.get("allowed", True)]
    if not violations:
        return []
    return [
        LeakageFlag(
            code="FIREWALL_VIOLATION",
            severity="critical",
            message=(
                f"{len(violations)} PIT firewall violations observed. "
                "This indicates the firewall was run in non-strict mode and the adapter "
                "requested data from beyond as_of. Hard leakage."
            ),
            statistic=float(len(violations)),
        )
    ]


def _impossible_accuracy_flag(
    decisions: pd.DataFrame, returns_by_ticker: dict[str, pd.Series]
) -> list[LeakageFlag]:
    if decisions is None or decisions.empty:
        return []
    # Directional accuracy: did the (LONG/SHORT) call match the realized sign of the next return?
    correct = 0
    total = 0
    for _, row in decisions.iterrows():
        direction = row["direction"]
        if direction == "FLAT":
            continue
        ret = row.get("ret_next")
        if pd.isna(ret) or ret == 0:
            continue
        sign_call = 1 if direction == "LONG" else -1
        sign_real = 1 if ret > 0 else -1
        total += 1
        if sign_call == sign_real:
            correct += 1
    if total < 30:
        # Too few non-FLAT calls; we cannot make a reliable claim.
        return []
    hit_rate = correct / total
    # Binomial test: H0: p=0.5; one-sided alternative p>0.5.
    pval = stats.binomtest(correct, total, p=0.5, alternative="greater").pvalue
    flags: list[LeakageFlag] = []
    if hit_rate > 0.85 and pval < 1e-4:
        flags.append(
            LeakageFlag(
                code="IMPOSSIBLE_ACCURACY",
                severity="critical",
                message=(
                    f"Directional hit rate {hit_rate:.3f} on {total} non-FLAT calls "
                    f"is implausibly high (binomial p={pval:.2e}). Investigate for leakage."
                ),
                statistic=hit_rate,
                pvalue=pval,
            )
        )
    elif hit_rate > 0.65 and pval < 1e-3:
        flags.append(
            LeakageFlag(
                code="SUSPICIOUS_ACCURACY",
                severity="warn",
                message=(
                    f"Directional hit rate {hit_rate:.3f} on {total} non-FLAT calls "
                    f"is high (binomial p={pval:.2e}). Worth a manual review."
                ),
                statistic=hit_rate,
                pvalue=pval,
            )
        )
    return flags


def _corp_action_proximity_flag(
    decisions: pd.DataFrame, corp_actions_by_ticker: dict[str, pd.DataFrame]
) -> list[LeakageFlag]:
    if decisions is None or decisions.empty:
        return []
    # Window: [-2, 0] trading days around ex_date. We use calendar days as a cheap proxy
    # since corporate actions are typically known a few business days ahead.
    window_days = 2
    near_total = 0
    near_correct = 0
    far_total = 0
    far_correct = 0
    decisions = decisions.copy()
    decisions["as_of"] = pd.to_datetime(decisions["as_of"])

    near_dates: dict[str, set[pd.Timestamp]] = {}
    for ticker, df in corp_actions_by_ticker.items():
        if df is None or df.empty:
            continue
        s = set()
        for _, row in df.iterrows():
            ex = pd.Timestamp(row["ex_date"])
            for off in range(-window_days, 1):
                s.add(ex + pd.Timedelta(days=off))
        near_dates[ticker] = s

    if not near_dates:
        return []

    for _, row in decisions.iterrows():
        direction = row["direction"]
        if direction == "FLAT":
            continue
        ret = row.get("ret_next")
        if pd.isna(ret) or ret == 0:
            continue
        sign_call = 1 if direction == "LONG" else -1
        sign_real = 1 if ret > 0 else -1
        hit = sign_call == sign_real
        is_near = row["as_of"] in near_dates.get(row["ticker"], set())
        if is_near:
            near_total += 1
            near_correct += int(hit)
        else:
            far_total += 1
            far_correct += int(hit)

    if near_total < 10 or far_total < 30:
        return []
    near_rate = near_correct / near_total
    far_rate = far_correct / far_total
    # One-sided z-test for difference of two proportions
    pooled = (near_correct + far_correct) / (near_total + far_total)
    se = np.sqrt(pooled * (1 - pooled) * (1 / near_total + 1 / far_total))
    if se <= 0:
        return []
    z = (near_rate - far_rate) / se
    pval = 1 - stats.norm.cdf(z)
    if pval < 0.01 and near_rate - far_rate > 0.1:
        return [
            LeakageFlag(
                code="CORP_ACTION_PROXIMITY",
                severity="warn",
                message=(
                    f"Directional accuracy near corporate-action ex-dates "
                    f"({near_rate:.3f} on {near_total}) significantly exceeds the elsewhere rate "
                    f"({far_rate:.3f} on {far_total}). Potential leakage via retroactive adjustment."
                ),
                statistic=float(near_rate - far_rate),
                pvalue=float(pval),
            )
        ]
    return []


def detect_leakage(
    *,
    audit_events: list[dict],
    decisions: pd.DataFrame,
    returns_by_ticker: dict[str, pd.Series] | None = None,
    corp_actions_by_ticker: dict[str, pd.DataFrame] | None = None,
) -> LeakageReport:
    flags: list[LeakageFlag] = []
    flags.extend(_firewall_violation_flag(audit_events))
    flags.extend(_impossible_accuracy_flag(decisions, returns_by_ticker or {}))
    flags.extend(_corp_action_proximity_flag(decisions, corp_actions_by_ticker or {}))
    return LeakageReport(flags=flags)
