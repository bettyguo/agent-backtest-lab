"""CLI smoke tests via Click's CliRunner."""
from __future__ import annotations

import json

from click.testing import CliRunner

from abl import __version__
from abl.cli.main import cli
from abl.config import DISCLAIMER_LINE


def test_cli_version():
    runner = CliRunner()
    res = runner.invoke(cli, ["version"])
    assert res.exit_code == 0
    assert __version__ in res.output


def test_cli_baselines_prints_disclaimer():
    runner = CliRunner()
    res = runner.invoke(cli, ["baselines"])
    assert res.exit_code == 0
    assert DISCLAIMER_LINE in res.output
    assert "buy_and_hold" in res.output


def test_cli_evaluate_writes_reports(tmp_path):
    runner = CliRunner()
    out = tmp_path / "run1"
    res = runner.invoke(
        cli,
        [
            "evaluate",
            "buy_and_hold",
            "--window", "2020-01-06..2021-01-05",
            "--out", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert (out / "report.md").exists()
    assert (out / "report.json").exists()
    md = (out / "report.md").read_text(encoding="utf-8")
    assert DISCLAIMER_LINE in md
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["adapter_name"] == "buy_and_hold"


def test_cli_evaluate_disclaimer_present_in_invocation(tmp_path):
    """The disclaimer banner must be in stdout on every evaluate invocation."""
    runner = CliRunner()
    out = tmp_path / "run2"
    res = runner.invoke(
        cli,
        ["evaluate", "buy_and_hold", "--window", "2020-01-06..2020-12-31", "--out", str(out), "--quiet"],
    )
    assert res.exit_code == 0, res.output
    # Even with --quiet, the disclaimer is present.
    assert DISCLAIMER_LINE in res.output


def test_cli_scorecard_reprint(tmp_path):
    runner = CliRunner()
    out = tmp_path / "run3"
    r1 = runner.invoke(cli, ["evaluate", "buy_and_hold", "--window", "2020-01-06..2020-12-31", "--out", str(out)])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(cli, ["scorecard", str(out)])
    assert r2.exit_code == 0
    assert "Scorecard" in r2.output


def test_cli_leakage_check(tmp_path):
    runner = CliRunner()
    out = tmp_path / "run4"
    r1 = runner.invoke(cli, ["evaluate", "buy_and_hold", "--window", "2020-01-06..2020-12-31", "--out", str(out)])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(cli, ["leakage-check", str(out)])
    assert r2.exit_code == 0
    # Buy-and-hold should not raise any leakage flag.
    assert "No leakage flags raised" in r2.output
