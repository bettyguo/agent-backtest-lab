"""Self-contained HTML renderer for the Scorecard.

Produces a single .html file with inline CSS and (optionally) inline base64-encoded
PNGs. No external dependencies at render time; the resulting file opens in any browser
and renders identically online or offline.

We deliberately keep the HTML simple — same disclaimer block, same not-advice framing,
same fields as the Markdown variant. The Markdown version remains the canonical artifact.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np

from abl.config import ATTRIBUTION, DISCLAIMER_BLOCK, DISCLAIMER_LINE
from abl.scorecard.report import Scorecard

_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; color: #222; line-height: 1.5; }
  h1 { border-bottom: 2px solid #1f4e8c; padding-bottom: 0.5rem; }
  h2 { color: #1f4e8c; margin-top: 2rem; }
  .disclaimer { background: #fff8e1; border-left: 4px solid #f0ad4e;
                padding: 0.8rem 1rem; margin: 1rem 0; font-size: 0.9rem; white-space: pre-wrap; }
  .banner { padding: 0.8rem 1rem; border-radius: 4px; font-weight: 600; margin: 1rem 0; }
  .banner-green { background: #e6f4ea; color: #1b5e20; border-left: 4px solid #2e7d32; }
  .banner-amber { background: #fff8e1; color: #6d4c00; border-left: 4px solid #f0ad4e; }
  .banner-red   { background: #fdecea; color: #b71c1c; border-left: 4px solid #c62828; }
  table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; }
  th, td { border-bottom: 1px solid #eee; padding: 0.4rem 0.6rem; text-align: left; }
  th { background: #fafafa; font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .flag { padding: 0.6rem 0.8rem; margin: 0.3rem 0; border-radius: 4px; }
  .flag-critical { background: #fdecea; border-left: 4px solid #c62828; }
  .flag-warn     { background: #fff8e1; border-left: 4px solid #f0ad4e; }
  .flag-info     { background: #e3f2fd; border-left: 4px solid #1976d2; }
  .footnote { color: #666; font-size: 0.85rem; margin-top: 1.5rem; padding-top: 0.5rem;
              border-top: 1px solid #eee; }
  .attribution { color: #777; font-size: 0.85rem; font-style: italic; }
  img.plot { display: block; max-width: 100%; height: auto; margin: 0.5rem 0; }
</style>
"""


def _esc(s: str) -> str:
    return html.escape(str(s))


def _embed_png(path: Path) -> str:
    """Read a PNG file and embed it as a base64 data URI."""
    if not path.exists():
        return ""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f'<img class="plot" src="data:image/png;base64,{b64}" alt="{_esc(path.name)}" />'


def _banner_class(sc: Scorecard) -> tuple[str, str]:
    rh_critical = any(f.severity == "critical" for f in sc.reward_hacking_flags)
    if sc.leakage_flags:
        return "banner banner-red", "LEAKAGE FLAGS RAISED — interpret with extreme caution."
    if rh_critical:
        return "banner banner-red", "REWARD-HACKING FLAG (critical) — strategy looks tuned to the IS window."
    if sc.overfitting_flag is not None and sc.overfitting_flag.severity == "critical":
        return "banner banner-red", "OVERFITTING FLAG (critical) — the in-sample-best is reliably bad OOS."
    if sc.overfitting_flag is not None or sc.reward_hacking_flags:
        return "banner banner-amber", "WARN-LEVEL FLAGS — see the Flags section below."
    if (sc.dsr is not None and sc.dsr < 0.5) or sc.psr < 0.5:
        return "banner banner-amber", "Caution — DSR/PSR below 0.5; not statistically distinguishable from the null."
    return "banner banner-green", "No critical flags raised. Note: the absence of a flag is not a guarantee."


def render_html(sc: Scorecard, *, plots_dir: str | Path | None = None) -> str:
    """Produce a single-file HTML report.

    If `plots_dir` is provided, look for `equity.png` / `drawdown.png` and embed them
    inline as base64 data URIs.
    """
    banner_cls, banner_text = _banner_class(sc)
    parts: list[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8" />')
    parts.append(f"<title>Scorecard — {_esc(sc.adapter_name)}</title>")
    parts.append(_CSS)
    parts.append("</head><body>")
    parts.append(f"<h1>Scorecard — {_esc(sc.adapter_name)}</h1>")
    parts.append(f'<div class="disclaimer">{_esc(DISCLAIMER_BLOCK)}</div>')
    parts.append(f'<div class="{banner_cls}">{_esc(banner_text)}</div>')

    parts.append("<h2>Run</h2><ul>")
    parts.append(f"<li><b>Universe:</b> {_esc(sc.universe_name)} ({sc.universe_size} tickers)</li>")
    parts.append(f"<li><b>Window:</b> {_esc(sc.window_start)} → {_esc(sc.window_end)} ({sc.n_trading_days} trading days)</li>")
    parts.append(f"<li><b>Cost model:</b> {_esc(sc.cost_model)}</li>")
    parts.append(f"<li><b>Annualization:</b> {sc.annualization}</li>")
    parts.append(f"<li><b>Generated:</b> {_esc(sc.generated_at_utc)}</li>")
    parts.append("</ul>")

    parts.append("<h2>Net-of-cost performance</h2>")
    parts.append("<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>")
    parts.append(f"<tr><td>Net total return</td><td class='num'>{sc.net_return_total:+.4%}</td></tr>")
    parts.append(f"<tr><td>Net annualized return</td><td class='num'>{sc.net_return_annualized:+.4%}</td></tr>")
    parts.append(f"<tr><td>Sharpe (annualized)</td><td class='num'>{sc.sharpe_annualized:+.3f}</td></tr>")
    ci_method = (
        f"HAC q={sc.sharpe_hac_lags}, η={sc.sharpe_hac_eta:.2f}"
        if sc.sharpe_ci_method == "hac" else "IID Gaussian"
    )
    parts.append(
        f"<tr><td>Sharpe 95% CI (annualized) — {ci_method}</td>"
        f"<td class='num'>[{sc.sharpe_ci_95_low:+.3f}, {sc.sharpe_ci_95_high:+.3f}]</td></tr>"
    )
    parts.append(f"<tr><td>PSR (vs SR=0)</td><td class='num'>{sc.psr:.4f}</td></tr>")
    if sc.dsr is not None:
        parts.append(
            f"<tr><td><b>DSR</b> (n_trials={sc.n_trials_reported}, SR₀={sc.sr_0:.3f})</td>"
            f"<td class='num'><b>{sc.dsr:.4f}</b></td></tr>"
        )
    if sc.pbo is not None:
        parts.append(f"<tr><td>PBO</td><td class='num'>{sc.pbo:.3f}</td></tr>")
    if np.isfinite(sc.sortino_annualized):
        parts.append(
            f"<tr><td>Sortino (annualized, MAR=0)</td>"
            f"<td class='num'>{sc.sortino_annualized:+.3f}</td></tr>"
        )
    if sc.information_ratio_annualized is not None:
        parts.append(
            f"<tr><td>Information Ratio (vs buy-and-hold, ann.)</td>"
            f"<td class='num'>{sc.information_ratio_annualized:+.3f}</td></tr>"
        )
    parts.append("</tbody></table>")

    parts.append("<h2>Drawdown</h2>")
    parts.append("<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>")
    parts.append(f"<tr><td>Max drawdown</td><td class='num'>{sc.max_drawdown:+.4%}</td></tr>")
    parts.append(f"<tr><td>Longest underwater stretch</td><td class='num'>{sc.longest_underwater_days} trading days</td></tr>")
    if np.isfinite(sc.calmar_ratio):
        parts.append(f"<tr><td>Calmar ratio</td><td class='num'>{sc.calmar_ratio:+.3f}</td></tr>")
    parts.append("</tbody></table>")

    # Optional embedded plots
    if plots_dir is not None:
        p = Path(plots_dir)
        eq = _embed_png(p / "equity.png")
        dd = _embed_png(p / "drawdown.png")
        if eq or dd:
            parts.append("<h2>Charts</h2>")
            if eq:
                parts.append(eq)
            if dd:
                parts.append(dd)

    if sc.ticker_breakdown:
        parts.append("<h2>Per-ticker breakdown</h2>")
        parts.append("<table><thead><tr><th>Ticker</th><th>Decisions</th><th>Non-FLAT</th>"
                     "<th>Hit rate</th><th>Mean contribution</th><th>Sharpe (ann.)</th></tr></thead><tbody>")
        for r in sc.ticker_breakdown:
            hr = f"{r.hit_rate:.3f}" if np.isfinite(r.hit_rate) else "—"
            mc = f"{r.mean_ret_contribution:+.4%}" if np.isfinite(r.mean_ret_contribution) else "—"
            sr = f"{r.sharpe_annualized:+.3f}" if np.isfinite(r.sharpe_annualized) else "—"
            parts.append(
                f"<tr><td>{_esc(r.ticker)}</td><td class='num'>{r.n_decisions}</td>"
                f"<td class='num'>{r.n_non_flat}</td><td class='num'>{hr}</td>"
                f"<td class='num'>{mc}</td><td class='num'>{sr}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append("<h2>Baselines (same window, same cost model)</h2>")
    parts.append("<table><thead><tr><th>Baseline</th><th>Net total return</th><th>Sharpe (ann.)</th></tr></thead><tbody>")
    if sc.baselines:
        for b in sc.baselines:
            parts.append(
                f"<tr><td>{_esc(b.name)}</td>"
                f"<td class='num'>{b.net_return_total:+.4%}</td>"
                f"<td class='num'>{b.sharpe_annualized:+.3f}</td></tr>"
            )
    else:
        parts.append("<tr><td colspan='3'><i>none provided</i></td></tr>")
    parts.append("</tbody></table>")

    parts.append("<h2>Calibration</h2>")
    if sc.confidence_emitted:
        if sc.ece is not None:
            parts.append(f"<p><b>ECE</b> (10 bins): {sc.ece:.4f}</p>")
        if sc.conformal_empirical_coverage is not None:
            parts.append(
                f"<p>Split conformal (rolling window, α={sc.conformal_alpha_nominal}): "
                f"empirical coverage {sc.conformal_empirical_coverage:.3f}</p>"
            )
    else:
        parts.append(
            "<p><i>Not evaluable: the adapter did not emit a confidence. "
            "We do not fabricate a probability where the framework provided none.</i></p>"
        )

    # Flags
    parts.append("<h2>Flags</h2>")
    flag_blocks = []
    for f in sc.leakage_flags:
        cls = f"flag flag-{f.severity}"
        flag_blocks.append(
            f'<div class="{cls}"><b>[LEAKAGE / {_esc(f.severity.upper())}] {_esc(f.code)}</b> — {_esc(f.message)}</div>'
        )
    if sc.overfitting_flag is not None:
        of = sc.overfitting_flag
        cls = f"flag flag-{of.severity}"
        flag_blocks.append(
            f'<div class="{cls}"><b>[OVERFITTING / {_esc(of.severity.upper())}] {_esc(of.code)}</b> — {_esc(of.message)}</div>'
        )
    for f in sc.reward_hacking_flags:
        cls = f"flag flag-{f.severity}"
        flag_blocks.append(
            f'<div class="{cls}"><b>[REWARD-HACKING / {_esc(f.severity.upper())}] {_esc(f.code)}</b> — {_esc(f.message)}</div>'
        )
    for f in sc.universe_flags:
        cls = f"flag flag-{f.severity}"
        flag_blocks.append(
            f'<div class="{cls}"><b>[UNIVERSE / {_esc(f.severity.upper())}] {_esc(f.code)}</b> — {_esc(f.message)}</div>'
        )
    if not flag_blocks:
        parts.append("<p><i>No flags raised.</i></p>")
    else:
        parts.extend(flag_blocks)

    parts.append(f'<p class="footnote">{_esc(DISCLAIMER_LINE)}</p>')
    parts.append(f'<p class="attribution">{_esc(ATTRIBUTION)}</p>')
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"
