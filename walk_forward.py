"""Walk-forward evaluation for delphi-quant.

The pre-reg specifies rolling 24-month train / 1-month test, no overlap, with
parameters frozen per fold. The v0.1 backtester defined `walk_forward_split`
but the evaluation path never used it: `run_backtest` evaluated the full
sample, which leaks the entire history into the reported Sharpe. This module
wires the split into the evaluation path.

What "train" means for the v0.1 baselines: the three pre-registered strategies
have fixed, pre-registered parameters (no fitting), so the train window is used
only to (a) warm up the signal (momentum needs ~13 months of history before it
emits a position) and (b) prove the frozen-parameter discipline end to end. The
function accepts a `fit_fn` hook so v0.2 strategies that actually estimate
parameters can select them on the train window and freeze them for the test
window. When `fit_fn` is None the strategy's pre-registered defaults are used.

OOS aggregation: per-fold daily net returns are concatenated into a single OOS
return stream and metrics are computed once on that stream. This is the honest
way to aggregate; averaging per-fold Sharpes over short 1-month folds is noisy
and upward-biased toward folds with tiny denominators.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from backtester import BacktestConfig, apply_costs, compute_metrics, walk_forward_split

# A strategy maps a price frame to a weights frame.
StrategyFn = Callable[[pd.DataFrame], pd.DataFrame]
# A fit hook selects (frozen) params on a train frame and returns a StrategyFn.
FitFn = Callable[[pd.DataFrame], StrategyFn]


def gapped_walk_forward_split(
    prices: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 1,
    gap_months: int = 3,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Walk-forward splits with a purge gap between train end and test start.

    Each fold: train on [start, cursor], skip `gap_months`, then test on
    [cursor + gap, cursor + gap + test_months]. The gap relocates the test
    window forward (it does not shrink a fixed 1-month window to nothing), so
    the regime-stability gate actually exercises out-of-sample days that are a
    full quarter removed from the training data.
    """
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    start = prices.index[0]
    end = prices.index[-1]
    cursor = start + pd.DateOffset(months=train_months)
    while True:
        test_start = cursor + pd.DateOffset(months=gap_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        train = prices.loc[start:cursor]
        test = prices.loc[test_start:test_end]
        if len(train) > 100 and len(test) >= 5:
            splits.append((train, test))
        cursor += pd.DateOffset(months=test_months)
    return splits


def _warmup_aware_test_weights(
    strategy_fn: StrategyFn,
    train: pd.DataFrame,
    test: pd.DataFrame,
    full_prices: pd.DataFrame | None = None,
    cfg: BacktestConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute test-window net returns using prior history for warm-up only.

    The strategy is evaluated on a continuous price frame that runs from the
    train start through the end of the test window, so signals that need history
    (momentum, reversal) are warmed up causally. Only the test window's returns
    are kept. This preserves the no-look-ahead rule: a weight on a given day
    still depends only on prices up to that day.

    When `full_prices` is supplied (used by the regime-gap gate), the warm-up
    history is the continuous slice [train start, test end] of the full series,
    so a purge gap between train and test does not punch a hole in the causal
    signal history. The gap still relocates the test window a quarter forward;
    it does not let the strategy peek across the gap.
    """
    cfg = cfg or BacktestConfig()
    if full_prices is not None:
        combined = full_prices.loc[train.index[0] : test.index[-1]]
    else:
        combined = pd.concat([train, test.loc[~test.index.isin(train.index)]])
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()

    weights = strategy_fn(combined)
    returns = combined.pct_change(fill_method=None)
    net = apply_costs(weights, returns, cfg)

    test_idx = test.index
    net_test = net.loc[net.index.isin(test_idx)]
    weights_test = weights.loc[weights.index.isin(test_idx)]
    return net_test, weights_test


def run_walk_forward(
    prices: pd.DataFrame,
    strategy_fn: StrategyFn,
    cfg: BacktestConfig | None = None,
    train_months: int = 24,
    test_months: int = 1,
    fit_fn: FitFn | None = None,
    train_test_gap_months: int = 0,
) -> dict:
    """Run a strategy walk-forward and aggregate OOS metrics.

    Args:
        prices: T x N adjusted-close frame.
        strategy_fn: default (pre-registered) strategy used when fit_fn is None.
        cfg: backtest config (costs etc).
        train_months / test_months: rolling window sizes (pre-reg: 24 / 1).
        fit_fn: optional hook to select frozen params on each train window.
        train_test_gap_months: purge gap between train end and test start. The
            pre-reg regime-stability gate uses 3 months here.

    Returns:
        dict with aggregated OOS metrics, the per-fold OOS Sharpe list, and the
        number of folds.
    """
    cfg = cfg or BacktestConfig()
    if train_test_gap_months > 0:
        splits = gapped_walk_forward_split(
            prices,
            train_months=train_months,
            test_months=test_months,
            gap_months=train_test_gap_months,
        )
    else:
        splits = walk_forward_split(prices, train_months=train_months, test_months=test_months)

    oos_streams: list[pd.Series] = []
    oos_weight_frames: list[pd.DataFrame] = []
    per_fold_sharpe: list[float] = []
    n_folds_used = 0

    # With a purge gap, the warm-up needs continuous causal history across the
    # gap, so pass the full price frame; without a gap, train and test are
    # already contiguous.
    full_for_warmup = prices if train_test_gap_months > 0 else None

    for train, test in splits:
        active_fn = fit_fn(train) if fit_fn is not None else strategy_fn
        net_test, weights_test = _warmup_aware_test_weights(
            active_fn, train, test, full_prices=full_for_warmup, cfg=cfg
        )
        if len(net_test) == 0:
            continue

        fold_metrics = compute_metrics(net_test, weights_test, cfg)
        per_fold_sharpe.append(fold_metrics["sharpe"])
        oos_streams.append(net_test)
        oos_weight_frames.append(weights_test)
        n_folds_used += 1

    if not oos_streams:
        return {"n_folds": 0, "sharpe": 0.0, "ann_return": 0.0, "max_dd": 0.0, "n_days": 0}

    oos_returns = pd.concat(oos_streams).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_weights = pd.concat(oos_weight_frames).sort_index()
    oos_weights = oos_weights[~oos_weights.index.duplicated(keep="first")]

    metrics = compute_metrics(oos_returns, oos_weights, cfg)
    metrics["n_folds"] = n_folds_used
    metrics["n_days"] = len(oos_returns)
    metrics["per_fold_sharpe_mean"] = float(pd.Series(per_fold_sharpe).mean())
    metrics["per_fold_sharpe_std"] = float(pd.Series(per_fold_sharpe).std(ddof=1)) if len(per_fold_sharpe) > 1 else 0.0
    metrics["start"] = str(oos_returns.index[0].date())
    metrics["end"] = str(oos_returns.index[-1].date())
    return metrics
