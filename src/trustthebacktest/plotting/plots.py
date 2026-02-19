from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from trustthebacktest.overfitting.pbo import PBOResult


def plot_lambda_hist(result: PBOResult, bins: int = 30) -> None:
    """Histogram of lambda values with a vertical line at 0."""
    plt.figure()
    plt.hist(result.lambdas, bins=bins)
    plt.axvline(0.0)
    plt.title(f"Lambda distribution (PBO={result.pbo:.2f})")
    plt.xlabel("lambda (logit of relative rank)")
    plt.ylabel("count")
    plt.tight_layout()


def plot_rank_hist(result: PBOResult, bins: int | None = None) -> None:
    """Histogram of the winner's out-of-sample rank (1=best)."""
    if bins is None:
        bins = min(30, result.n_strategies)

    plt.figure()
    plt.hist(result.ranks, bins=bins)
    plt.title("Winner rank in test (1=best)")
    plt.xlabel("rank")
    plt.ylabel("count")
    plt.tight_layout()
