"""Look-ahead / data-leakage detection.

Two flavors:
1. Hard guarantee: any access via PITView with `requested_date > as_of` raises
   FirewallViolation. See `abl.data.firewall`.
2. Best-effort detection: post-hoc heuristics over the audit log + the decisions
   table to flag suspicious patterns. See `abl.leakage.detector`.

The post-hoc detectors exist because (a) Python cannot prevent a wrapped framework
from making its own out-of-band data calls, and (b) even within our firewall, some
leakage failure modes are subtle enough that the user might benefit from a
statistical sanity check (e.g. impossibly good calls clustered around known
data-revision dates).
"""
from __future__ import annotations

from abl.leakage.audit import parse_audit_events
from abl.leakage.detector import (
    LeakageFlag,
    LeakageReport,
    detect_leakage,
)

__all__ = ["LeakageFlag", "LeakageReport", "detect_leakage", "parse_audit_events"]
