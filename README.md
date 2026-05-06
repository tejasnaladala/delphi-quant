# delphi-quant

Multi-stage verification pipeline for systematic equity strategies, with a
delphi-style pre-registration discipline (every hypothesis logged before any
backtest runs).

**Lead**: Tejas Naladala (UW, ECE + Applied Math '28)
**Status**: v0.1, baseline pre-registered, autonomous loop ready

## Honest scope

This is not a pre-built alpha. It is a credible, audited research framework that
can iterate strategies overnight and cleanly separate "this looks like alpha"
from "this is a multiple-comparisons artifact." The realistic v0.1 goal is to
produce a small set of candidate strategies that survive strict OOS plus
multiple-comparison correction. Real-money deployment requires several months
of additional out-of-sample paper trading.

## v0.1 baseline results (pre-registered)

S&P large-cap universe (104 names), 2010-01-04 to 2025-12-30, daily bars.
TC 10 bps + slippage 5 bps per side. Equal-weight, no leverage.

| Strategy | Sharpe | Ann return | Max DD | Pre-reg verdict |
|---|---|---|---|---|
| Buy-and-hold | 0.83 | 14.8% | -33.2% | PASS (sanity) |
| TS momentum (12-1, top 20, monthly) | 0.77 | 15.2% | -35.6% | PASS (above 0.7 target) |
| XS mean reversion (5-day, bottom 20, weekly) | 0.26 | 5.4% | -40.2% | FAIL (below 0.3 dead threshold) |

XS mean reversion appears dead on this universe in this regime, consistent with
the literature on post-2010 alpha decay in short-horizon reversal.

## Repo layout

```
PRE_REGISTRATION.md     Locked hypotheses + success criteria + search-budget rules
backtester.py           Strict daily-bar backtester (TC, slippage, walk-forward)
strategies.py           Pre-registered baseline strategies
run_strategy.py         Single-run driver, emits METRIC lines for /autoresearch
benchmark.sh            Entry point for the autonomous /autoresearch loop
fetch_data.py           One-time data fetch (yfinance, cached to parquet)
data/                   Cached parquet, gitignored
results/                Per-run JSON, gitignored
```

## Quick start

```bash
pip install yfinance pandas numpy scipy pyarrow
python fetch_data.py
python run_strategy.py --strategy buy_and_hold
python run_strategy.py --strategy time_series_momentum
python run_strategy.py --strategy cross_sectional_mean_reversion
```

## Autonomous iteration via /autoresearch

The `benchmark.sh` script is compatible with the autoresearch-claude-code
skill. Open a Claude Code session here and run:

```
/autoresearch search for strategies that beat the time_series_momentum baseline
(Sharpe 0.77 OOS) on the audited S&P large-cap harness. Hold the backtester,
universe, TC, and slippage constants. Search over: signal definition (momentum
window 21 to 252 days, skip 0 to 21, top_n 5 to 50), additional features
(volatility scaling, sector neutralization, beta hedge), rebalance cadence
(weekly to quarterly). Metric: Sharpe net of TC. Multiple-comparisons gate:
Holm-Bonferroni at family-wise alpha 0.05. Pre-reg gate: any candidate with
Sharpe > 1.0 must be re-run on the 2024-2025 hold-out before being accepted.
```

## Pre-registration discipline

Every hypothesis is committed to `PRE_REGISTRATION.md` BEFORE any backtest is
run. Deviations are logged as deviations, not silently fixed. This is the same
rule applied to the Parchment Labs Sobol pilot (which violated its own
consistency check and was publicly logged) and the maze-rl-baselines harness
audit (which caught a survivorship-style filter and re-ran the affected
experiments transparently).

## Known limitations (v0.1)

1. **Survivorship bias**: current S&P constituents only, not the historical list.
   v0.2 will use the historical constituent list.
2. **No regime detection**: assumes static relationship across 2010-2025.
3. **No risk model**: no factor exposure controls, no covariance shrinkage.
4. **No microstructure**: daily bars, no intraday execution.
5. **No alternative data**: just price + volume.

These are deliberate v0.1 choices to keep the framework legible. v0.2+ will
relax them in a controlled order.

## License

MIT
