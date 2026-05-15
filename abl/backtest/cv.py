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
from itertools import combinations

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


@dataclass(frozen=True)
class CombinatorialSplit:
    """One split from a combinatorial purged K-fold scheme.

    `test_groups` is the tuple of group indices that comprise the test set; the test_idx
    is their union (sorted). `train_idx` is everything outside the purge+embargo window
    around each test group.
    """

    test_groups: tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray


def combinatorial_purged_kfold_splits(
    n: int,
    n_groups: int,
    n_test_groups: int,
    *,
    label_window: int = 1,
    embargo_pct: float = 0.01,
) -> list[CombinatorialSplit]:
    """Combinatorial Purged K-Fold CV (López de Prado 2018, Chapter 12).

    Partitions the chronological index [0, n) into `n_groups` equal-sized contiguous
    groups. For each of the C(n_groups, n_test_groups) ways of choosing groups to
    leave out, returns a split with:

    - `test_idx`: union of the chosen groups' indices.
    - `train_idx`: everything not in the test_idx and not in the purge+embargo window
      around any test group.

    The number of distinct "paths" (full out-of-sample series) recoverable from these
    splits is C(n_groups - 1, n_test_groups - 1). This is the López de Prado
    generalization of purged K-fold that allows path-level out-of-sample uncertainty.

    Parameters
    ----------
    n : int
        Number of chronologically-ordered observations.
    n_groups : int
        Total number of equal-sized groups to partition n into. Must satisfy n >= n_groups.
    n_test_groups : int
        Number of groups to hold out as test per split. Must satisfy 1 <= n_test_groups < n_groups.
    label_window, embargo_pct : as in `purged_kfold_splits`.

    Returns
    -------
    list[CombinatorialSplit]
        Length is C(n_groups, n_test_groups).

    Source
    ------
    López de Prado, M. (2018). *Advances in Financial Machine Learning*, Wiley, Chapter 12.
    """
    if n_groups < 2 or n_groups > n:
        raise ValueError(f"n_groups must satisfy 2 <= n_groups <= n; got {n_groups}, n={n}")
    if n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError(
            f"n_test_groups must satisfy 1 <= n_test_groups < n_groups; got {n_test_groups}"
        )
    if label_window < 0:
        raise ValueError("label_window must be >= 0")
    if not (0.0 <= embargo_pct < 1.0):
        raise ValueError("embargo_pct must be in [0, 1)")

    embargo = int(round(embargo_pct * n))
    bounds = np.linspace(0, n, n_groups + 1, dtype=int)
    splits: list[CombinatorialSplit] = []
    for combo in combinations(range(n_groups), n_test_groups):
        test_mask = np.zeros(n, dtype=bool)
        purge_mask = np.zeros(n, dtype=bool)
        for g in combo:
            g_lo, g_hi = bounds[g], bounds[g + 1]
            test_mask[g_lo:g_hi] = True
            purge_lo = max(0, g_lo - label_window)
            purge_hi = min(n, g_hi + label_window + embargo)
            purge_mask[purge_lo:purge_hi] = True
        train_mask = ~purge_mask & ~test_mask
        splits.append(
            CombinatorialSplit(
                test_groups=combo,
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
            )
        )
    return splits


def n_recoverable_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of distinct full-OOS paths recoverable from combinatorial purged K-fold.

    Per López de Prado (2018), this is C(n_groups - 1, n_test_groups - 1). A higher
    path count gives the user more independent OOS realizations of the strategy curve
    for variance estimation.
    """
    from math import comb

    if n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("require 1 <= n_test_groups < n_groups")
    return comb(n_groups - 1, n_test_groups - 1)
