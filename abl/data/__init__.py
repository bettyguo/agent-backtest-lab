"""Point-in-time data layer and the leakage firewall.

The single most important promise of this library is in this subpackage: when an
adapter receives a PITView, it cannot see any datum with date > as_of, and the
adjusted OHLCV series is reconstructed using only corporate actions with
ex_date <= as_of. The firewall logs every access.

See `abl.leakage` for the post-hoc detectors that run on the audit log.
"""
from __future__ import annotations

from abl.data.firewall import AccessEvent, Firewall, FirewallViolation
from abl.data.fixture_synth import generate_synthetic_fixture
from abl.data.loaders import (
    DataStore,
    load_fixture,
)
from abl.data.pit_view import PITView

__all__ = [
    "PITView",
    "Firewall",
    "AccessEvent",
    "FirewallViolation",
    "DataStore",
    "load_fixture",
    "generate_synthetic_fixture",
]
