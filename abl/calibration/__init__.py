"""Calibration analysis for agent confidence outputs.

- `expected_calibration_error` and `reliability_diagram_data` — Guo et al. (2017)
  ICML-style ECE for predicted probabilities.
- `split_conformal_set` and `rolling_split_conformal` — distribution-free coverage
  guarantees per Vovk/Gammerman/Shafer and Angelopoulos & Bates.

Honest framing
--------------
If the wrapped framework does not emit a confidence (most LLM trading agents don't),
the calibration module reports "not evaluable" rather than fabricating one. This is
deliberate. Our scorecard prints that absence as a visible field.
"""
from __future__ import annotations

from abl.calibration.conformal import (
    rolling_split_conformal_coverage,
    split_conformal_quantile,
)
from abl.calibration.reliability import (
    ReliabilityBin,
    expected_calibration_error,
    reliability_diagram_data,
)

__all__ = [
    "ReliabilityBin",
    "expected_calibration_error",
    "reliability_diagram_data",
    "split_conformal_quantile",
    "rolling_split_conformal_coverage",
]
