"""Standardized flag types for the scorecard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["info", "warn", "critical"]


@dataclass(frozen=True)
class UniverseFlag:
    """Concerns about the declared universe: survivorship, PIT correctness, etc."""

    code: str
    severity: Severity
    message: str
