"""Statistical tests for delphi-quant.

Two things live here:

1. A Sharpe-ratio significance test. The pre-reg evaluates strategies on
   Sharpe net of cost, so the per-strategy hypothesis test is "is this Sharpe
   distinguishable from zero". We use the standard large-sample result that the
   t-statistic for an annualized Sharpe SR over T daily observations is
   approximately SR * sqrt(T / trading_days_per_year), i.e. SR scaled back to
   per-period units times sqrt(T). The two-sided p-value comes from the normal
   tail. This is the Lo (2002) first-order approximation; it ignores the
   higher-moment correction, which is documented as a v0.1 limitation.

2. Holm-Bonferroni step-down multiple-comparison correction over the family of
   evaluated strategies. The pre-reg locks family-wise alpha at 0.05. Holm is
   uniformly more powerful than plain Bonferroni and still controls FWER under
   arbitrary dependence, which is the honest default when strategy p-values are
   correlated (they share a universe and a backtester).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def sharpe_pvalue(sharpe_annual: float, n_days: int, trading_days_per_year: int = 252) -> float:
    """Two-sided p-value that an annualized Sharpe differs from zero.

    The annualized Sharpe is SR_daily * sqrt(P) where P = trading_days_per_year.
    The t-statistic for SR_daily over T observations is SR_daily * sqrt(T).
    Substituting SR_daily = sharpe_annual / sqrt(P) gives
    t = sharpe_annual * sqrt(T / P). We return the two-sided normal p-value.
    """
    if n_days < 2 or trading_days_per_year <= 0:
        return 1.0
    t_stat = sharpe_annual * np.sqrt(n_days / trading_days_per_year)
    # Two-sided normal tail (large-sample approximation).
    return float(2.0 * stats.norm.sf(abs(t_stat)))


@dataclass(frozen=True)
class HolmResult:
    """Outcome of Holm-Bonferroni for one hypothesis in the family."""

    label: str
    p_value: float
    rank: int  # 1-based position in the ascending-p ordering
    adjusted_alpha: float  # alpha / (m - rank + 1)
    reject_null: bool  # significant after correction


def holm_bonferroni(pvalues: dict[str, float], family_alpha: float = 0.05) -> list[HolmResult]:
    """Holm-Bonferroni step-down correction.

    Args:
        pvalues: mapping of hypothesis label -> raw two-sided p-value.
        family_alpha: family-wise error rate to control (pre-reg: 0.05).

    Returns:
        One HolmResult per hypothesis, ordered ascending by raw p-value.

    Procedure: sort p ascending. Compare the k-th smallest (1-based) to
    alpha / (m - k + 1). Reject while the comparison holds; the first failure
    stops all further rejections (step-down monotonicity).
    """
    if not pvalues:
        return []
    m = len(pvalues)
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    results: list[HolmResult] = []
    still_rejecting = True
    for k, (label, p) in enumerate(ordered, start=1):
        adj_alpha = family_alpha / (m - k + 1)
        if still_rejecting and p <= adj_alpha:
            reject = True
        else:
            reject = False
            still_rejecting = False  # step-down: no later hypothesis can reject
        results.append(
            HolmResult(
                label=label,
                p_value=float(p),
                rank=k,
                adjusted_alpha=float(adj_alpha),
                reject_null=reject,
            )
        )
    return results
