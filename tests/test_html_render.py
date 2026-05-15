"""Tests for the HTML scorecard renderer."""
from __future__ import annotations

from abl.backtest.engine import WalkForwardEngine
from abl.baselines import buy_and_hold_adapter, naive_momentum_adapter
from abl.config import ATTRIBUTION, DISCLAIMER_BLOCK, DISCLAIMER_LINE  # noqa: F401
from abl.scorecard.html_render import render_html
from abl.scorecard.report import build_scorecard


def test_html_contains_disclaimer_and_attribution(synth_store, synth_universe, synth_window):
    engine = WalkForwardEngine(store=synth_store)
    primary = engine.run(buy_and_hold_adapter(), synth_universe, synth_window)
    baselines = [engine.run(naive_momentum_adapter(), synth_universe, synth_window)]
    sc = build_scorecard(primary=primary, baselines=baselines)
    html = render_html(sc)
    # The disclaimer line is HTML-escaped (apostrophe -> &#x27;). Check on a stable
    # substring rather than the raw line.
    assert "Not financial advice" in html
    assert "Not a trading system" in html
    # The block has line breaks; check on a stable phrase.
    assert "research and evaluation tool" in html
    assert "Backtest results are not predictive" in html
    # Attribution.
    assert "Betty Guo" in html or "Dongxin Guo" in html
    # Self-contained: must be a full HTML document.
    assert "<!doctype html>" in html.lower()
    assert "</html>" in html.lower()


def test_html_includes_baselines_and_breakdown(synth_store, synth_universe, synth_window):
    engine = WalkForwardEngine(store=synth_store)
    primary = engine.run(buy_and_hold_adapter(), synth_universe, synth_window)
    baselines = [engine.run(naive_momentum_adapter(), synth_universe, synth_window)]
    sc = build_scorecard(primary=primary, baselines=baselines)
    html = render_html(sc)
    assert "Per-ticker breakdown" in html
    assert "SYN-A" in html
    assert "naive_momentum" in html


def test_html_embeds_plot_when_provided(tmp_path, synth_store, synth_universe, synth_window):
    from abl.plots.renderers import plot_drawdown, plot_equity_curve
    engine = WalkForwardEngine(store=synth_store)
    primary = engine.run(buy_and_hold_adapter(), synth_universe, synth_window)
    baselines = [engine.run(naive_momentum_adapter(), synth_universe, synth_window)]
    sc = build_scorecard(primary=primary, baselines=baselines)
    plot_equity_curve(primary.daily_pnl_net, {"naive_momentum": baselines[0].daily_pnl_net},
                      tmp_path / "equity.png", primary_name="bah")
    plot_drawdown(primary.daily_pnl_net, tmp_path / "drawdown.png")
    html = render_html(sc, plots_dir=tmp_path)
    # base64-encoded image data URI must appear
    assert "data:image/png;base64," in html


def test_html_no_gross_returns_leak(synth_store, synth_universe, synth_window):
    """Same compliance assertion as the Markdown test: no gross-return rendering."""
    engine = WalkForwardEngine(store=synth_store)
    primary = engine.run(buy_and_hold_adapter(), synth_universe, synth_window)
    sc = build_scorecard(primary=primary, baselines=[])
    html = render_html(sc)
    assert "gross return" not in html.lower()
    assert "Gross" not in html
