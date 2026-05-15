"""Purged K-fold cross-validation with embargo, per López de Prado (2018), Ch. 7.

Source
------
Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018, Chapter 7.

Statement
---------
When labels are constructed from forward windows of length `label_window`, a naive
K-fold split leaks information into the training set whenever a training observation's
label window overlaps any test observation. The fix:

1. **Purge** training observations whose label window `[i, i + label_window]` overlaps
   the test fold `[test_start, test_end + label_window]`.
2. **Embargo** an additional `embargo_pct * n` observations immediately after each test
   fold to neutralize serial correlation that the purge alone cannot remove.

Assumptions
-----------
- Labels are constructed from a forward window of known constant length.
- Serial correlation decays within the embargo.

Fixture test (see tests/test_walkforward.py): construct a dataset with known label
overlap and assert no train index falls within [test_start - label_window, test_end + embargo].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedKFoldSplit:
    """One fold's train / test index arrays."""

    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_kfold_splits(
    n: int,
    k: int,
    *,
    label_window: int = 1,
    embargo_pct: float = 0.01,
) -> list[PurgedKFoldSplit]:
    """Produce K purged folds over `n` chronologically-ordered observations.

    Parameters
    ----------
    n : int
        Number of observations (rows in chronological order; index 0 is earliest).
    k : int
        Number of folds; must be >= 2 and <= n.
    label_window : int, default 1
        Length of the forward label window (e.g. 5 for a 5-day forward return).
    embargo_pct : float, default 0.01
        Fraction of total observations to embargo after each test fold (López de Prado's
        suggested order of magnitude is 0.01–0.02).

    Returns
    -------
    list[PurgedKFoldSplit]
        K folds. For each: `test_idx` is a contiguous block; `train_idx` is everything
        outside `[test_start - label_window, test_end + label_window + embargo)`.
    """
    if k < 2 or k > n:
        raise ValueError(f"k must satisfy 2 <= k <= n; got k={k}, n={n}")
    if label_window < 0:
        raise ValueError("label_window must be >= 0")
    if not (0.0 <= embargo_pct < 1.0):
        raise ValueError("embargo_pct must be in [0, 1)")

    embargo = int(round(embargo_pct * n))
    # Contiguous test folds
    fold_bounds = np.linspace(0, n, k + 1, dtype=int)
    splits: list[PurgedKFoldSplit] = []
    for i in range(k):
        test_start, test_end = fold_bounds[i], fold_bounds[i + 1]
        if test_end <= test_start:
            continue
        # Purge any training observation whose label window overlaps the test fold,
        # and embargo `embargo` rows right after the test fold.
        purge_lo = max(0, test_start - label_window)
        purge_hi = min(n, test_end + label_window + embargo)
        train_mask = np.ones(n, dtype=bool)
        train_mask[purge_lo:purge_hi] = False
        train_idx = np.where(train_mask)[0]
        test_idx = np.arange(test_start, test_end)
        splits.append(PurgedKFoldSplit(train_idx=train_idx, test_idx=test_idx))
    return splits
