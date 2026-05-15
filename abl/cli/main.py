"""`abl` CLI.

The not-advice disclaimer is printed on every invocation. `--quiet` silences progress
output but NEVER silences the disclaimer.
"""
from __future__ import annotations

import importlib
import json
from datetime import date
from pathlib import Path

import click

from abl import __version__
from abl.adapters.callable_adapter import CallableAdapter
from abl.backtest.engine import EngineConfig, WalkForwardEngine
from abl.baselines import (
    buy_and_hold_adapter,
    naive_momentum_adapter,
    random_baseline_adapter,
)
from abl.config import DEFAULT_ANNUALIZATION, DEFAULT_COST_BPS, DISCLAIMER_LINE
from abl.costs.models import ConstantBpsCost
from abl.data.loaders import load_fixture
from abl.scorecard.report import build_scorecard, render_json, render_markdown
from abl.types import Universe, Window


def _print_banner() -> None:
    # The disclaimer banner. Always printed. Not removable.
    click.secho(DISCLAIMER_LINE, fg="yellow", err=False)


def _resolve_adapter(spec: str):
    """Resolve adapter-spec strings into adapter instances."""
    s = spec.strip()
    if s in ("buy_and_hold", "bah"):
        return buy_and_hold_adapter()
    if s in ("naive_momentum", "naive", "momentum"):
        return naive_momentum_adapter()
    if s == "random":
        return random_baseline_adapter()
    if s == "tradingagents:default":
        from abl.adapters.tradingagents import TradingAgentsAdapter
        return TradingAgentsAdapter()
    if ":" in s:
        module_path, callable_name = s.split(":", 1)
        mod = importlib.import_module(module_path)
        fn = getattr(mod, callable_name)
        return CallableAdapter(fn=fn, name=callable_name)
    raise click.ClickException(
        f"Unknown adapter spec: {spec!r}. Use one of: buy_and_hold, naive_momentum, random, "
        "tradingagents:default, or module.path:callable."
    )


def _parse_window(s: str) -> Window:
    if ".." not in s:
        raise click.ClickException("--window must be START..END (YYYY-MM-DD..YYYY-MM-DD)")
    a, b = s.split("..", 1)
    return Window(start=date.fromisoformat(a), end=date.fromisoformat(b))


@click.group()
@click.version_option(__version__, prog_name="abl")
def cli() -> None:
    """agent-backtest-lab — audit your LLM trading agent."""


@cli.command()
@click.argument("adapter_spec", type=str)
@click.option("--window", required=True, type=str, help="YYYY-MM-DD..YYYY-MM-DD")
@click.option("--cost-bps", default=DEFAULT_COST_BPS, type=float, help="Per-side bps cost")
@click.option("--annualization", default=DEFAULT_ANNUALIZATION, type=int)
@click.option("--n-trials", default=1, type=int, help="Number of trials user reports having tried (for DSR)")
@click.option("--var-trial-sharpe", default=0.0, type=float, help="Cross-trial Sharpe variance (for DSR)")
@click.option("--out", default="abl_out", type=click.Path(file_okay=False), help="Output directory")
@click.option("--quiet/--no-quiet", default=False, help="Suppress progress chatter (does NOT silence the disclaimer)")
def evaluate(adapter_spec, window, cost_bps, annualization, n_trials, var_trial_sharpe, out, quiet):
    """Run the standard evaluation: adapter + three baselines + scorecard."""
    _print_banner()
    if not quiet:
        click.echo(f"[abl] adapter: {adapter_spec}")
        click.echo(f"[abl] window: {window}")
        click.echo(f"[abl] cost: {cost_bps} bps each side")

    adapter = _resolve_adapter(adapter_spec)
    win = _parse_window(window)
    store = load_fixture()
    universe = Universe(
        tickers=store.tickers(), name="bundled_synthetic", survivorship_verified=True,
        source_note="synthetic GBM fixture; no survivorship bias by construction"
    )
    cost = ConstantBpsCost(bps_each_side=cost_bps)
    engine = WalkForwardEngine(store=store, config=EngineConfig(cost_model=cost))

    primary = engine.run(adapter, universe, win)
    baselines = [
        engine.run(buy_and_hold_adapter(), universe, win),
        engine.run(naive_momentum_adapter(), universe, win),
        engine.run(random_baseline_adapter(), universe, win),
    ]

    sc = build_scorecard(
        primary=primary,
        baselines=baselines,
        n_trials_reported=n_trials,
        var_trial_sharpe=var_trial_sharpe,
        cost_model_name=f"constant_bps_{cost_bps:g}",
        annualization=annualization,
    )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render_markdown(sc), encoding="utf-8")
    (out_dir / "report.json").write_text(render_json(sc), encoding="utf-8")
    if not quiet:
        click.echo(f"[abl] wrote {out_dir / 'report.md'}")
        click.echo(f"[abl] wrote {out_dir / 'report.json'}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def scorecard(run_dir):
    """Re-print a previously generated scorecard."""
    _print_banner()
    p = Path(run_dir) / "report.md"
    if not p.exists():
        raise click.ClickException(f"no report.md in {run_dir}")
    click.echo(p.read_text(encoding="utf-8"))


@cli.command(name="leakage-check")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def leakage_check(run_dir):
    """Re-print the leakage section of a previously generated scorecard."""
    _print_banner()
    p = Path(run_dir) / "report.json"
    if not p.exists():
        raise click.ClickException(f"no report.json in {run_dir}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    flags = payload.get("leakage_flags") or []
    if not flags:
        click.echo("No leakage flags raised.")
        return
    for f in flags:
        click.echo(f"[{f['severity'].upper()}] {f['code']}: {f['message']}")


@cli.command(name="methods")
def methods():
    """Print docs/METHODS.md — the statistical-method reference."""
    _print_banner()
    p = Path(__file__).resolve().parents[2] / "docs" / "METHODS.md"
    if p.exists():
        click.echo(p.read_text(encoding="utf-8"))
    else:
        click.echo(
            "See docs/METHODS.md in the repository for the full statistical-method reference."
        )


@cli.command()
def baselines():
    """List the built-in baselines."""
    _print_banner()
    click.echo("buy_and_hold       LONG every ticker every day; passive benchmark.")
    click.echo("naive_momentum     LONG if trailing 20-day return positive; FLAT otherwise.")
    click.echo("random             Uniform LONG/FLAT/SHORT per (ticker, as_of); seeded.")


@cli.command()
def version():
    """Print the version string."""
    click.echo(__version__)


if __name__ == "__main__":
    cli(prog_name="abl")
