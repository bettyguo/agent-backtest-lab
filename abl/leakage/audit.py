"""Parser for the firewall audit log (list of dict events from a BacktestResult)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditSummary:
    total_accesses: int
    violations: int
    tickers_touched: int


def parse_audit_events(events: list[dict]) -> AuditSummary:
    if not events:
        return AuditSummary(total_accesses=0, violations=0, tickers_touched=0)
    violations = sum(1 for e in events if not e.get("allowed", True))
    tickers = {e.get("ticker") for e in events}
    tickers.discard("_universe_")
    return AuditSummary(total_accesses=len(events), violations=violations, tickers_touched=len(tickers))
