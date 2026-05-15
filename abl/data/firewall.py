"""The leakage firewall.

Every PIT data access goes through `Firewall.check()`. Any request for a datum with
`requested_date > as_of` raises `FirewallViolation`. Every access is recorded in an
audit log that the post-hoc leakage detectors consume.

A FirewallViolation in product code is a bug. In test code it is what the deliberately-leaky
fixture is supposed to trigger to prove the gate works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

AccessKind = Literal["bars", "corporate_actions", "metadata"]


class FirewallViolation(Exception):
    """Raised when a PIT access requests data with date > as_of."""


@dataclass(frozen=True)
class AccessEvent:
    """One audit-log entry."""

    as_of: date
    ticker: str
    kind: AccessKind
    requested_min_date: date
    requested_max_date: date
    allowed: bool


@dataclass
class Firewall:
    """Mediates every PIT access and accumulates the audit log."""

    strict: bool = True  # if False, log violations but do not raise (used only in fault tests)
    events: list[AccessEvent] = field(default_factory=list)

    def check(
        self,
        *,
        as_of: date,
        ticker: str,
        kind: AccessKind,
        requested_min_date: date,
        requested_max_date: date,
    ) -> None:
        """Validate an access. Logs the event whether allowed or denied."""
        allowed = requested_max_date <= as_of
        ev = AccessEvent(
            as_of=as_of,
            ticker=ticker,
            kind=kind,
            requested_min_date=requested_min_date,
            requested_max_date=requested_max_date,
            allowed=allowed,
        )
        self.events.append(ev)
        if not allowed and self.strict:
            raise FirewallViolation(
                f"PIT firewall: access to {ticker} {kind} requested up to "
                f"{requested_max_date} but as_of is {as_of}"
            )

    def violations(self) -> list[AccessEvent]:
        return [e for e in self.events if not e.allowed]

    def reset(self) -> None:
        self.events.clear()
