"""Reliability diagrams + Expected Calibration Error (ECE).

Source
------
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern
  Neural Networks. Proceedings of the 34th ICML, PMLR 70:1321-1330.
  arXiv: https://arxiv.org/abs/1706.04599
- Older lineage: Murphy, A. H. (1973). A New Vector Partition of the Probability Score.
  J. Applied Meteorology, 12, 595-600.

Statement
---------
Partition the [0, 1] interval into M equal-width bins. For each bin b:

    acc(b) = (1 / |b|) Σ_{i in b} 1[ŷ_i = y_i]
    conf(b) = (1 / |b|) Σ_{i in b} p̂_i

ECE = Σ_b (|b| / N) · |acc(b) - conf(b)|

A perfectly calibrated model has acc(b) ≈ conf(b) for every populated bin and ECE → 0.

Convention
----------
We use the binary version where `probs` is the predicted probability of the positive
class and `labels` is 0/1. Calibration in the multi-class setting (top-label vs all-class
ECE) is out of scope for this library — agent confidences are typically a scalar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lo: float
    hi: float
    count: int
    mean_confidence: float
    empirical_accuracy: float


def reliability_diagram_data(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Return per-bin reliability stats. Empty bins are omitted."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have the same shape")
    if probs.ndim != 1:
        raise ValueError("probs and labels must be 1-D")
    if probs.size == 0:
        return []
    if np.any((probs < 0) | (probs > 1) | ~np.isfinite(probs)):
        raise ValueError("probs must be in [0, 1] and finite")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be 0/1")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins; map p == 1.0 into the last bin
    bin_idx = np.minimum(np.searchsorted(edges, probs, side="right") - 1, n_bins - 1)
    bin_idx = np.maximum(bin_idx, 0)

    out: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        out.append(
            ReliabilityBin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                count=cnt,
                mean_confidence=float(probs[mask].mean()),
                empirical_accuracy=float(labels[mask].mean()),
            )
        )
    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error per Guo et al. (2017).

    ECE = Σ_b (|b| / N) · |acc(b) - conf(b)|.
    """
    bins = reliability_diagram_data(probs, labels, n_bins=n_bins)
    if not bins:
        return float("nan")
    n_total = sum(b.count for b in bins)
    return float(
        sum(b.count / n_total * abs(b.empirical_accuracy - b.mean_confidence) for b in bins)
    )
