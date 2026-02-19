from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CSCVSplits:
    """
    CSCV split generator:
    - Split the time series into S contiguous slices.
    - For each combination of K slices, treat them as training slices; the rest are test slices.
    """
    S: int
    K: int

    def combos(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        if not (0 < self.K < self.S):
            raise ValueError("Require 0 < K < S")

        all_idx = tuple(range(self.S))
        out: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        for train in itertools.combinations(all_idx, self.K):
            test = tuple(i for i in all_idx if i not in train)
            out.append((train, test))
        return out


def slice_indices(n_obs: int, S: int) -> List[np.ndarray]:
    """Split indices 0..n_obs-1 into S contiguous slices."""
    if S < 2:
        raise ValueError("S must be >= 2")
    idx = np.arange(n_obs)
    return [x for x in np.array_split(idx, S)]


def fold_returns(returns: np.ndarray, slices: Sequence[np.ndarray]) -> np.ndarray:
    """
    Return an object array of length S where each element is the returns for that slice.
    """
    r = np.asarray(returns, dtype=float)
    out = np.empty(len(slices), dtype=object)
    for i, sidx in enumerate(slices):
        out[i] = r[sidx]
    return out


def concat_slices(slice_series: np.ndarray, which: Iterable[int]) -> np.ndarray:
    """Concatenate slice return series for given slice indices."""
    parts = [slice_series[i] for i in which]
    return np.concatenate(parts) if parts else np.array([], dtype=float)

