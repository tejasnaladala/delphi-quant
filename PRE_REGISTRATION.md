# Pre-Registration: delphi-quant v0.1

**Date locked**: 2026-04-27
**Lead**: Tejas Naladala
**Repo state at lock**: see git commit `git rev-parse HEAD`

This document is a delphi-style pre-registration. It locks the hypotheses,
universe, evaluation methodology, and success criteria BEFORE any backtest is
run. Deviations from this document must be logged as deviations (with reason)
in `DEVIATION_LOG.md`, not silently fixed. The same rule we applied to the
Parchment Labs Sobol pilot when it violated its own consistency check, and the
same rule we applied to the maze-rl-baselines harness audit.

## 1. Universe

- **Asset class**: US equities, daily bars
- **Constituent list**: current S&P 500 (per Wikipedia at fetch time)
- **Time range**: 2010-01-01 to 2025-12-31
- **Bias acknowledged**: survivorship-biased (current constituents only).
  v0.2 will use the historical constituent list to remove this bias. All v0.1
  results will be re-run on v0.2 data and the deltas reported.

## 2. Evaluation harness

- **Backtester**: `backtester.py`, locked at this commit
- **Transaction costs**: 10 bps per side
- **Slippage**: 5 bps per side
- **Lag**: signals computed at close on day t execute at close on day t+1;
  returns realized on day t+2 (no look-ahead)
- **Position sizing**: equal-weight on selected names, capped at 1.0 gross
  exposure
- **Walk-forward CV**: rolling 24-month train / 1-month test, no overlap, frozen
  parameters per fold
- **Risk-free rate**: 4% annual (current 3-month T-bill approximation)

## 3. Pre-registered baseline strategies (v0.1)

All three strategies are committed in `strategies.py` BEFORE any backtest runs.
Their hypotheses and success criteria are locked here.

### H1: Buy-and-hold (sanity check)

- **Hypothesis**: equal-weight buy-and-hold on the S&P 500 universe should
  produce Sharpe in [0.4, 0.9] OOS over 2010-2025 (matches historical SPX).
- **Success**: in-range Sharpe + max drawdown in [-15%, -45%].
- **Failure**: anything outside those ranges indicates a backtester bug.

### H2: Time-series momentum (Jegadeesh & Titman 1993, Asness 1994)

- **Hypothesis**: top-20 by 12-month return (skip most recent month) outperforms
  buy-and-hold on Sharpe net of TC.
- **Pre-registered Sharpe target**: > 0.7 OOS net of TC.
- **Pre-registered failure mode**: Sharpe <= 0.5 OOS would indicate the
  strategy is dead in this regime.

### H3: Cross-sectional mean reversion (Lo & MacKinlay 1990, Jegadeesh 1990)

- **Hypothesis**: bottom-20 by 5-day return outperforms buy-and-hold on Sharpe
  net of TC.
- **Pre-registered Sharpe target**: > 0.5 OOS net of TC.
- **Pre-registered failure mode**: Sharpe <= 0.3 OOS would indicate the strategy
  is dead in this regime (consistent with the broad finding that pure XS
  reversal has been arbitraged away post-2010).

## 4. Search budget for v0.1 autonomous iteration

The `/autoresearch` loop has the following limits:

- **Max strategies evaluated**: 200
- **Max walk-forward folds per strategy**: as defined by the harness
- **Multiple-comparison correction**: Holm-Bonferroni at family-wise alpha = 0.05
  on the set of all evaluated strategies
- **Pre-reg gate**: any strategy with OOS Sharpe > 1.5 must be re-run on the
  hold-out 2024-2025 period (not used for any selection) to qualify as a
  candidate
- **Capital deployment gate**: candidate must additionally hold up under:
  1. 5x TC (50 bps per side) sensitivity test
  2. Re-run with universe restricted to S&P 100 (liquidity sanity check)
  3. Re-run with 3-month gap between train and test (regime stability check)

## 5. What is not allowed

- Looking at OOS metrics during strategy iteration (only IS metrics for
  hyperparameter selection)
- Adding hyperparameters not in the pre-reg without logging the addition as a
  deviation
- Running the same strategy with different random seeds and reporting only the
  best (the harness is deterministic; if randomness is added it must be averaged
  over a pre-declared seed set)
- Selecting the universe or time range based on what looks like it works
- Modifying the backtester after seeing results (any backtester change requires
  re-running all v0.1 baselines)

## 6. Reporting

- All strategy results, including failures and rejections, are saved to
  `results/` and committed
- Multiple-comparison-corrected significance is reported alongside raw p-values
- The full search history is committed (no cherry-picking)
- A v0.1 final report is generated whether or not any candidate qualifies

## 7. Honest scope

- v0.1 will not produce a strategy ready for real-money deployment
- v0.1 produces: a credible, pre-registered evaluation framework + a search
  trajectory + a list of candidates that pass strict OOS + multiple-comparison
  correction
- Real-money deployment requires v0.2 (survivorship-bias-corrected) +
  3-6 months of paper-trade OOS validation
