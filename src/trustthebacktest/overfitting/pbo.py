from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from trustthebacktest.metrics.performance import sharpe_ratio
from trustthebacktest.overfitting.cscv import CSCVSplits, concat_slices, fold_returns, slice_indices

MetricFn = Callable[[np.ndarray], float]


@dataclass
class PBOResult:
    pbo: float
    lambdas: np.ndarray
    ranks: np.ndarray
    n_strategies: int
    n_folds: int
    S: int
    K: int


def _relative_rank(rank: int, n: int) -> float:
    # maps rank 1..n to (0,1) avoiding 0/1
    return (rank - 0.5) / n


def estimate_pbo(
    strategy_returns: np.ndarray,
    S: int = 16,
    K: Optional[int] = None,
    metric: Optional[MetricFn] = None,
    periods_per_year: int = 252,
    seed: int = 42,
) -> PBOResult:
    """
    Estimate Probability of Backtest Overfitting (PBO) via CSCV.

    strategy_returns: (T, N) array of periodic simple returns
    S: number of slices
    K: number of training slices per fold (default: S//2)
    metric: function mapping 1D return series -> score (default: annualized Sharpe)
    """
    r = np.asarray(strategy_returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("strategy_returns must be 2D: (T, N)")
    T, N = r.shape
    if T < S:
        raise ValueError("Need at least T >= S observations")

    if K is None:
        K = S // 2
    if metric is None:
        metric = lambda x: sharpe_ratio(x, periods_per_year=periods_per_year)

    rng = np.random.default_rng(seed)

    # Split time into S contiguous slices, pre-store each strategy's slice return series
    slices = slice_indices(T, S)
    per_strategy_slices = [fold_returns(r[:, j], slices) for j in range(N)]

    combos = CSCVSplits(S=S, K=K).combos()
    lambdas, ranks = [], []

    for train_slices, test_slices in combos:
        # Train scores
        train_scores = np.array(
            [metric(concat_slices(per_strategy_slices[j], train_slices)) for j in range(N)],
            dtype=float,
        )

        best = np.nanmax(train_scores)
        candidates = np.where(train_scores == best)[0]
        winner = int(rng.choice(candidates)) if candidates.size > 1 else int(candidates[0])

        # Test scores + rank winner
        test_scores = np.array(
            [metric(concat_slices(per_strategy_slices[j], test_slices)) for j in range(N)],
            dtype=float,
        )
        safe = np.where(np.isnan(test_scores), -np.inf, test_scores)
        order = np.argsort(-safe)  # descending
        winner_rank = int(np.where(order == winner)[0][0] + 1)  # 1..N

        rr = _relative_rank(winner_rank, N)
        lam = float(np.log(rr / (1.0 - rr)))  # logit

        ranks.append(winner_rank)
        lambdas.append(lam)

    lambdas = np.asarray(lambdas, dtype=float)
    ranks = np.asarray(ranks, dtype=int)
    pbo_val = float(np.mean(lambdas > 0.0))

    return PBOResult(
        pbo=pbo_val,
        lambdas=lambdas,
        ranks=ranks,
        n_strategies=N,
        n_folds=len(combos),
        S=S,
        K=K,
    )

