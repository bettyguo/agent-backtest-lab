"""Scorecard data model + Markdown / JSON renderers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from abl.calibration.conformal import rolling_split_conformal_coverage
from abl.calibration.reliability import expected_calibration_error
from abl.config import (
    ATTRIBUTION,
    DEFAULT_ANNUALIZATION,
    DEFAULT_CONFORMAL_ALPHA,
    DISCLAIMER_BLOCK,
    DISCLAIMER_LINE,
)
from abl.leakage.detector import LeakageFlag, detect_leakage
from abl.leakage.reward_hacking import RewardHackingFlag, detect_reward_hacking
from abl.multipletest.dsr import deflated_sharpe
from abl.multipletest.hac import hac_sharpe_ci
from abl.multipletest.psr import probabilistic_sharpe, sharpe_ratio
from abl.multipletest.risk_metrics import risk_metrics_summary
from abl.overfitting.cscv import OverfittingFlag
from abl.scorecard.breakdown import per_ticker_breakdown
from abl.scorecard.drawdown import drawdown_stats
from abl.scorecard.flags import UniverseFlag
from abl.types import BacktestResult


@dataclass
class BaselineRow:
    name: str
    net_return_total: float
    sharpe_annualized: float


@dataclass
class Scorecard:
    adapter_name: str
    universe_name: str
    universe_size: int
    window_start: str
    window_end: str
    n_trading_days: int
    cost_model: str
    annualization: int

    net_return_total: float
    net_return_annualized: float
    sharpe_observed: float
    sharpe_annualized: float
    sharpe_ci_95_low: float
    sharpe_ci_95_high: float
    sharpe_ci_method: str  # "iid" or "hac"
    sharpe_hac_eta: float | None  # HAC serial-correlation correction factor; None for IID
    sharpe_hac_lags: int | None
    psr: float
    dsr: float | None
    sr_0: float | None
    n_trials_reported: int

    pbo: float | None

    # Drawdown metrics — always computed from the net equity curve
    max_drawdown: float
    longest_underwater_days: int
    calmar_ratio: float

    # Downside / benchmark-relative metrics
    sortino_annualized: float
    information_ratio_annualized: float | None  # vs buy-and-hold; None if not computed

    # Per-ticker breakdown — empty if requested off; always populated by default
    ticker_breakdown: list = field(default_factory=list)

    baselines: list[BaselineRow] = field(default_factory=list)

    ece: float | None = None
    conformal_empirical_coverage: float | None = None
    conformal_alpha_nominal: float | None = None
    confidence_emitted: bool = False

    leakage_flags: list[LeakageFlag] = field(default_factory=list)
    overfitting_flag: OverfittingFlag | None = None
    reward_hacking_flags: list[RewardHackingFlag] = field(default_factory=list)
    universe_flags: list[UniverseFlag] = field(default_factory=list)

    generated_at_utc: str = ""
    disclaimer: str = DISCLAIMER_LINE
    attribution: str = ATTRIBUTION


def _sharpe_ci(stats_obj, alpha: float = 0.05) -> tuple[float, float]:
    """Normal-approximation CI for the observed-cadence Sharpe ratio.

    Uses the IID-Gaussian variance Var(ŜR) ≈ (1 + 0.5·ŜR²)/T (Lo 2002 Eq. 6 under
    the IID assumption). The CI is constructed on the observed-cadence Sharpe; the
    caller scales the bounds by √annualization.

    HONEST CAVEAT: this estimator does NOT correct for serial correlation in the
    return series. Real strategy returns are autocorrelated and the IID standard
    error UNDERSTATES variance — that is, the printed CI is narrower than the truly
    appropriate one. For strategies with material autocorrelation, the user should
    interpret the CI as a lower-bound on uncertainty. PSR and DSR pick up the
    higher-moment corrections; a HAC / Newey-West SE for the CI itself is a planned
    extension (see roadmap).

    Reference: Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial
    Analysts Journal*, 58(4), 36-52.
    """
    z = norm.ppf(1.0 - alpha / 2)
    se_obs = np.sqrt((1.0 + 0.5 * stats_obj.sharpe_observed ** 2) / stats_obj.n_obs)
    lo_obs = stats_obj.sharpe_observed - z * se_obs
    hi_obs = stats_obj.sharpe_observed + z * se_obs
    return lo_obs, hi_obs


def build_scorecard(
    *,
    primary: BacktestResult,
    baselines: list[BacktestResult] | None = None,
    n_trials_reported: int = 1,
    var_trial_sharpe: float = 0.0,
    pbo_result: dict | None = None,
    universe_flags: list[UniverseFlag] | None = None,
    annualization: int = DEFAULT_ANNUALIZATION,
    cost_model_name: str = "constant_bps_5bps",
    sharpe_ci_method: str = "hac",
) -> Scorecard:
    """Assemble a Scorecard from a primary BacktestResult and (always) baselines.

    `sharpe_ci_method`:
      - "hac" (default): use Newey-West HAC SE for the Sharpe CI. Handles autocorrelation.
      - "iid": IID-Gaussian SE. Faster, but understates variance under autocorrelation.
    """
    if sharpe_ci_method not in ("hac", "iid"):
        raise ValueError(f"sharpe_ci_method must be 'hac' or 'iid', got {sharpe_ci_method!r}")
    rets = primary.daily_pnl_net.dropna().to_numpy()
    if rets.size < 2:
        raise ValueError("primary result has fewer than 2 net-return observations")
    stats = sharpe_ratio(rets, annualization=annualization)
    scale = float(np.sqrt(annualization))
    if sharpe_ci_method == "hac":
        hac_result = hac_sharpe_ci(rets, alpha=0.05, annualization=annualization)
        sr_lo_obs = hac_result.ci_low_obs
        sr_hi_obs = hac_result.ci_high_obs
        sharpe_hac_eta: float | None = hac_result.eta_q
        sharpe_hac_lags: int | None = hac_result.q_lags
    else:
        sr_lo_obs, sr_hi_obs = _sharpe_ci(stats)
        sharpe_hac_eta = None
        sharpe_hac_lags = None
    psr = probabilistic_sharpe(rets, sr_benchmark=0.0, annualization=annualization)
    dd_stats = drawdown_stats(rets, annualization=annualization)
    ticker_rows = per_ticker_breakdown(primary.decisions, annualization=annualization)

    # Look for a buy-and-hold baseline to use as the IR benchmark; align by date.
    bah_baseline = None
    for b in baselines or []:
        if b.adapter_name == "buy_and_hold":
            bah_baseline = b
            break
    if bah_baseline is not None:
        common_idx = primary.daily_pnl_net.index.intersection(bah_baseline.daily_pnl_net.index)
        bench_aligned = bah_baseline.daily_pnl_net.reindex(common_idx).dropna().to_numpy()
        prim_aligned = primary.daily_pnl_net.reindex(common_idx).dropna().to_numpy()
        # Reindex to the same final length
        m = min(len(bench_aligned), len(prim_aligned))
        risk_metrics = risk_metrics_summary(
            prim_aligned[:m], bench_aligned[:m], annualization=annualization
        )
    else:
        risk_metrics = risk_metrics_summary(rets, None, annualization=annualization)

    dsr: float | None = None
    sr_0: float | None = None
    if n_trials_reported > 1 or var_trial_sharpe > 0:
        d = deflated_sharpe(
            rets, n_trials=n_trials_reported, var_trial_sharpe=var_trial_sharpe, annualization=annualization
        )
        dsr = d["dsr"]
        sr_0 = d["sr_0"]

    net_return_total = float(np.prod(1.0 + rets) - 1.0)
    n_days = int(rets.size)
    net_return_annualized = float((1.0 + net_return_total) ** (annualization / n_days) - 1.0) if n_days > 0 else float("nan")

    baseline_rows: list[BaselineRow] = []
    for b in baselines or []:
        b_rets = b.daily_pnl_net.dropna().to_numpy()
        if b_rets.size < 2:
            continue
        b_stats = sharpe_ratio(b_rets, annualization=annualization)
        b_ret = float(np.prod(1.0 + b_rets) - 1.0)
        baseline_rows.append(
            BaselineRow(name=b.adapter_name, net_return_total=b_ret, sharpe_annualized=b_stats.sharpe_annualized)
        )

    # Calibration: only if confidence emitted
    ece: float | None = None
    conformal_emp: float | None = None
    confidence_emitted = False
    if "confidence" in primary.decisions.columns:
        conf = primary.decisions["confidence"].to_numpy()
        has_conf = ~pd.isna(conf)
        if has_conf.any():
            confidence_emitted = True
            # Binary label: did the LONG/SHORT call match next-day return sign?
            ret_next = primary.decisions["ret_next"].to_numpy()
            direction = primary.decisions["direction"].to_numpy()
            mask = (direction != "FLAT") & has_conf & ~pd.isna(ret_next) & (ret_next != 0)
            if mask.sum() >= 50:
                pred = primary.decisions["confidence"].to_numpy()[mask].astype(float)
                sign_call = np.where(direction[mask] == "LONG", 1, -1)
                sign_real = np.where(ret_next[mask] > 0, 1, -1)
                correct = (sign_call == sign_real).astype(int)
                ece = expected_calibration_error(pred, correct, n_bins=10)
                # Use nonconformity score = 1 - confidence-in-correct-label; rolling conformal.
                ncf = 1.0 - np.where(correct == 1, pred, 1.0 - pred)
                if ncf.size > 100:
                    out = rolling_split_conformal_coverage(
                        ncf, alpha=DEFAULT_CONFORMAL_ALPHA, window=max(50, ncf.size // 4)
                    )
                    conformal_emp = float(out["empirical_coverage"])

    leakage_report = detect_leakage(
        audit_events=primary.audit_events, decisions=primary.decisions
    )
    reward_hacking = detect_reward_hacking(
        returns=primary.daily_pnl_net,
        decisions=primary.decisions,
        annualization=annualization,
    )

    return Scorecard(
        adapter_name=primary.adapter_name,
        universe_name=primary.universe.name,
        universe_size=len(primary.universe.tickers),
        window_start=primary.window.start.isoformat(),
        window_end=primary.window.end.isoformat(),
        n_trading_days=n_days,
        cost_model=cost_model_name,
        annualization=annualization,
        net_return_total=net_return_total,
        net_return_annualized=net_return_annualized,
        sharpe_observed=stats.sharpe_observed,
        sharpe_annualized=stats.sharpe_annualized,
        sharpe_ci_95_low=sr_lo_obs * scale,
        sharpe_ci_95_high=sr_hi_obs * scale,
        sharpe_ci_method=sharpe_ci_method,
        sharpe_hac_eta=sharpe_hac_eta,
        sharpe_hac_lags=sharpe_hac_lags,
        psr=psr,
        dsr=dsr,
        sr_0=sr_0,
        n_trials_reported=n_trials_reported,
        pbo=(pbo_result["pbo"] if pbo_result is not None else None),
        max_drawdown=dd_stats.max_drawdown,
        longest_underwater_days=dd_stats.longest_underwater_days,
        calmar_ratio=dd_stats.calmar_ratio,
        sortino_annualized=risk_metrics.sortino_annualized,
        information_ratio_annualized=(
            risk_metrics.information_ratio_annualized
            if np.isfinite(risk_metrics.information_ratio_annualized) else None
        ),
        ticker_breakdown=ticker_rows,
        baselines=baseline_rows,
        ece=ece,
        conformal_empirical_coverage=conformal_emp,
        conformal_alpha_nominal=DEFAULT_CONFORMAL_ALPHA if confidence_emitted else None,
        confidence_emitted=confidence_emitted,
        leakage_flags=leakage_report.flags,
        overfitting_flag=(pbo_result["flag"] if pbo_result is not None else None),
        reward_hacking_flags=reward_hacking,
        universe_flags=list(universe_flags or _default_universe_flags(primary)),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _default_universe_flags(result: BacktestResult) -> list[UniverseFlag]:
    flags: list[UniverseFlag] = []
    if not result.universe.survivorship_verified:
        flags.append(
            UniverseFlag(
                code="SURVIVORSHIP_UNVERIFIED",
                severity="warn",
                message=(
                    "The declared universe is not marked survivorship-verified. Public-data "
                    "universes (yfinance, stooq) silently drop delistings, which inflates "
                    "backtest returns. Interpret with caution."
                ),
            )
        )
    return flags


def render_markdown(sc: Scorecard) -> str:
    """Render a Scorecard to Markdown. Disclaimer block at top, table of metrics, flags
    section, baselines table, attribution at bottom. No gross-only numbers."""
    rh_critical = any(f.severity == "critical" for f in sc.reward_hacking_flags)
    if sc.leakage_flags:
        banner = "🔴 **LEAKAGE FLAGS RAISED** — interpret with extreme caution."
    elif rh_critical:
        banner = "🔴 **REWARD-HACKING FLAG (critical)** — strategy looks tuned to the IS window."
    elif sc.overfitting_flag is not None and sc.overfitting_flag.severity == "critical":
        banner = "🔴 **OVERFITTING FLAG (critical)** — the in-sample-best is reliably bad OOS."
    elif sc.overfitting_flag is not None or sc.reward_hacking_flags:
        banner = "🟡 **WARN-LEVEL FLAGS** — see the Flags section below."
    elif (sc.dsr is not None and sc.dsr < 0.5) or sc.psr < 0.5:
        banner = "🟡 **Caution** — DSR/PSR below 0.5; the observed Sharpe is not statistically distinguishable from the multiple-testing null."
    else:
        banner = "🟢 No critical flags raised. *Note: the absence of a flag is not a guarantee.*"

    lines: list[str] = []
    lines.append(f"# Scorecard — {sc.adapter_name}")
    lines.append("")
    lines.append("> " + DISCLAIMER_BLOCK.replace("\n", "\n> "))
    lines.append("")
    lines.append(banner)
    lines.append("")
    lines.append(f"- **Universe:** {sc.universe_name} ({sc.universe_size} tickers)")
    lines.append(f"- **Window:** {sc.window_start} → {sc.window_end} ({sc.n_trading_days} trading days)")
    lines.append(f"- **Cost model:** {sc.cost_model}")
    lines.append(f"- **Annualization:** {sc.annualization}")
    lines.append(f"- **Generated:** {sc.generated_at_utc}")
    lines.append("")
    lines.append("## Net-of-cost performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Net total return | {sc.net_return_total:+.4%} |")
    lines.append(f"| Net annualized return | {sc.net_return_annualized:+.4%} |")
    lines.append(f"| Sharpe (annualized) | {sc.sharpe_annualized:+.3f} |")
    ci_method_label = (
        f"HAC ({sc.sharpe_ci_method.upper()}, q={sc.sharpe_hac_lags}, η={sc.sharpe_hac_eta:.2f})"
        if sc.sharpe_ci_method == "hac"
        else "IID Gaussian"
    )
    lines.append(
        f"| Sharpe 95% CI (annualized)¹ — {ci_method_label} | "
        f"[{sc.sharpe_ci_95_low:+.3f}, {sc.sharpe_ci_95_high:+.3f}] |"
    )
    lines.append(f"| PSR (vs SR=0) | {sc.psr:.4f} |")
    if sc.dsr is not None:
        lines.append(f"| **DSR** (deflated, N_trials={sc.n_trials_reported}, SR₀={sc.sr_0:.3f}) | **{sc.dsr:.4f}** |")
    else:
        lines.append("| DSR | not computed (n_trials_reported=1, var_trial_sharpe=0) |")
    if sc.pbo is not None:
        lines.append(f"| PBO (Probability of Backtest Overfitting) | {sc.pbo:.3f} |")
    if np.isfinite(sc.sortino_annualized):
        lines.append(f"| Sortino (annualized, MAR=0) | {sc.sortino_annualized:+.3f} |")
    if sc.information_ratio_annualized is not None:
        lines.append(
            f"| Information Ratio (vs buy-and-hold, ann.) | {sc.information_ratio_annualized:+.3f} |"
        )

    lines.append("")
    lines.append("## Drawdown")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Max drawdown | {sc.max_drawdown:+.4%} |")
    lines.append(f"| Longest underwater stretch | {sc.longest_underwater_days} trading days |")
    if np.isfinite(sc.calmar_ratio):
        lines.append(f"| Calmar ratio (ann. ret / |max DD|) | {sc.calmar_ratio:+.3f} |")
    else:
        lines.append("| Calmar ratio | n/a (no drawdown observed) |")

    if sc.ticker_breakdown:
        lines.append("")
        lines.append("## Per-ticker breakdown")
        lines.append("")
        lines.append(
            "_A real edge spreads across the universe; a fragile edge concentrates in a "
            "single ticker. Hit-rate is over non-FLAT calls only._"
        )
        lines.append("")
        lines.append("| Ticker | Decisions | Non-FLAT | Hit rate | Mean contribution | Sharpe (ann.) |")
        lines.append("|---|---|---|---|---|---|")
        for row in sc.ticker_breakdown:
            hr = f"{row.hit_rate:.3f}" if np.isfinite(row.hit_rate) else "—"
            mc = f"{row.mean_ret_contribution:+.4%}" if np.isfinite(row.mean_ret_contribution) else "—"
            sr = f"{row.sharpe_annualized:+.3f}" if np.isfinite(row.sharpe_annualized) else "—"
            lines.append(
                f"| {row.ticker} | {row.n_decisions} | {row.n_non_flat} | {hr} | {mc} | {sr} |"
            )

    lines.append("")
    lines.append("## Baselines (same window, same cost model)")
    lines.append("")
    lines.append("| Baseline | Net total return | Sharpe (ann.) |")
    lines.append("|---|---|---|")
    for b in sc.baselines:
        lines.append(f"| {b.name} | {b.net_return_total:+.4%} | {b.sharpe_annualized:+.3f} |")
    if not sc.baselines:
        lines.append("| _none provided_ | — | — |")

    lines.append("")
    lines.append("## Calibration")
    lines.append("")
    if sc.confidence_emitted:
        lines.append(f"- Expected Calibration Error (ECE, 10 bins): **{sc.ece:.4f}**" if sc.ece is not None else "- ECE: not computed")
        if sc.conformal_empirical_coverage is not None:
            lines.append(
                f"- Split conformal (rolling window, α={sc.conformal_alpha_nominal}): "
                f"empirical coverage {sc.conformal_empirical_coverage:.3f}"
            )
    else:
        lines.append(
            "- Not evaluable: the adapter did not emit a confidence. This is honest — we do not "
            "fabricate a probability where the framework provided none."
        )

    lines.append("")
    lines.append("## Flags")
    lines.append("")
    if sc.leakage_flags:
        lines.append("### Leakage")
        for f in sc.leakage_flags:
            lines.append(f"- **[{f.severity.upper()}] {f.code}** — {f.message}")
    if sc.overfitting_flag is not None:
        of = sc.overfitting_flag
        lines.append("### Overfitting")
        lines.append(f"- **[{of.severity.upper()}] {of.code}** — {of.message}")
    if sc.reward_hacking_flags:
        lines.append("### Reward hacking / window overfitting")
        for f in sc.reward_hacking_flags:
            lines.append(f"- **[{f.severity.upper()}] {f.code}** — {f.message}")
    if sc.universe_flags:
        lines.append("### Universe")
        for f in sc.universe_flags:
            lines.append(f"- **[{f.severity.upper()}] {f.code}** — {f.message}")
    if not (sc.leakage_flags or sc.overfitting_flag or sc.universe_flags):
        lines.append("- No flags raised.")

    lines.append("")
    lines.append("---")
    lines.append("")
    if sc.sharpe_ci_method == "hac":
        lines.append(
            f"¹ The Sharpe 95% CI uses the HAC / Newey-West SE (Lo 2002 with Bartlett kernel, "
            f"q={sc.sharpe_hac_lags} lags, autocorrelation correction factor η={sc.sharpe_hac_eta:.2f}). "
            "η > 1 indicates the IID CI would have been overconfident; η < 1 the opposite. "
            "Use `build_scorecard(..., sharpe_ci_method='iid')` for the IID-Gaussian variant."
        )
    else:
        lines.append(
            "¹ The Sharpe 95% CI uses the IID-Gaussian normal approximation (Lo 2002). "
            "It does NOT correct for serial correlation. Use "
            "`build_scorecard(..., sharpe_ci_method='hac')` to switch to the Newey-West variant."
        )
    lines.append("")
    lines.append(f"_{DISCLAIMER_LINE}_")
    lines.append("")
    lines.append(f"_{ATTRIBUTION}_")
    return "\n".join(lines) + "\n"


def render_json(sc: Scorecard) -> str:
    """Render a Scorecard to JSON. Same fields as Markdown; flags serialized as dicts."""
    def _flag_to_dict(f):
        if f is None:
            return None
        if hasattr(f, "__dataclass_fields__"):
            return asdict(f)
        return {"code": getattr(f, "code", None), "severity": getattr(f, "severity", None),
                "message": getattr(f, "message", None)}

    payload = asdict(sc)
    payload["leakage_flags"] = [_flag_to_dict(f) for f in sc.leakage_flags]
    payload["overfitting_flag"] = _flag_to_dict(sc.overfitting_flag)
    payload["reward_hacking_flags"] = [_flag_to_dict(f) for f in sc.reward_hacking_flags]
    payload["universe_flags"] = [_flag_to_dict(f) for f in sc.universe_flags]
    return json.dumps(payload, indent=2, default=str)
