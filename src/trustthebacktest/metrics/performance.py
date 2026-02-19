from __future__ import annotations

import numpy as np


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """
    Annualized Sharpe ratio computed from periodic returns.

    returns: 1D array of periodic simple returns (e.g., daily).
    rf: risk-free rate per period (same frequency as returns).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return np.nan

    excess = r - rf
    vol = excess.std(ddof=1)
    if vol == 0:
        return np.nan

    return np.sqrt(periods_per_year) * excess.mean() / vol


def cumulative_return(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return np.nan
    return float(np.prod(1.0 + r) - 1.0)


def max_drawdown(returns: np.ndarray) -> float:
    """
    Max drawdown from a returns series (simple returns).
    Returns a negative number (e.g., -0.25 for -25%).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return np.nan

    equity = np.cumprod(1.0 + r)
    peaks = np.maximum.accumulate(equity)
    dd = equity / peaks - 1.0
    return float(dd.min())
