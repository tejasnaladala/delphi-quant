# delphi-quant

Multi-stage verification pipeline for systematic equity strategies, with a
delphi-style pre-registration discipline (every hypothesis logged before any
backtest runs).

**Lead**: Tejas Naladala (UW, ECE + Applied Math '28)
**Status**: v0.1 pipeline wired end to end (walk-forward OOS, multiple-comparison
correction, deployment gates, structured rejection logging)

## Honest scope

This is not a pre-built alpha. It is an audited research framework that iterates
strategies overnight and separates "this looks like alpha" from "this is a
multiple-comparisons artifact." The v0.1 goal is a small set of candidate
strategies that survive strict out-of-sample evaluation plus multiple-comparison
correction plus capital-deployment stress tests. A candidate is not
deployment-ready: real money requires v0.2 (survivorship-bias-corrected data)
and several months of paper-trade out-of-sample validation.

## v0.1 baseline results (pre-registered)

S&P large-cap universe (103 names; 104 attempted, 1 failed download),
2010-01-04 to 2025-12-30, daily bars. TC 10 bps + slippage 5 bps per side.
Equal-weight, no leverage. Numbers are **walk-forward out-of-sample** (24-month
train / 1-month test, frozen parameters per fold), aggregated over the
concatenated per-fold OOS return streams. Reproduce with `python run_pipeline.py`;
the run also writes `REPORT_v0.1.md` and `results/rejection_log.jsonl`.

| Strategy | OOS Sharpe | Raw p | Holm sig (FWER 0.05) | Pre-reg verdict |
|---|---|---|---|---|
| Buy-and-hold | 0.90 | 0.001 | yes | PASS (sanity, in [0.4, 0.9]) |
| TS momentum (12-1, top 20, monthly) | 0.84 | 0.002 | yes | CANDIDATE (target 0.7, cleared all gates) |
| XS mean reversion (5-day, bottom 20, weekly) | 0.29 | 0.285 | no | FAIL (below 0.3 dead threshold) |

TS momentum is the one v0.1 candidate: OOS Sharpe 0.84 clears the pre-registered
0.7 target, stays significant after Holm-Bonferroni, and survives all three
deployment gates (5x TC: 0.67; S&P-100 liquidity subset: 0.94; 3-month
train/test regime gap: 0.80). It is a candidate, not a deployment.

XS mean reversion is dead on this universe in this regime: OOS Sharpe 0.29 sits
below the pre-registered 0.3 floor and is not statistically distinguishable from
zero (raw p 0.285). This matches the literature on post-2010 alpha decay in
short-horizon reversal.

These numbers were refreshed against a live yfinance pull on 2026-06-15 after
the evaluation path moved from full-sample to the pre-registered walk-forward
method. The data-snapshot drift and the method change are both logged in
`DEVIATION_LOG.md`, as the pre-reg requires.

## Repo layout

```
PRE_REGISTRATION.md     Locked hypotheses + success criteria + search-budget rules
DEVIATION_LOG.md        Logged departures from the pre-reg (numbers, methods)
backtester.py           Strict daily-bar backtester (TC, slippage, metrics)
walk_forward.py         Walk-forward OOS runner (frozen params, optional purge gap)
strategies.py           Pre-registered baseline strategies
stats.py                Sharpe significance test + Holm-Bonferroni correction
gates.py                Capital-deployment gates (5x TC, liquidity, regime gap)
rejection_log.py        Structured per-strategy check/threshold/verdict records
pipeline.py             Multi-stage verification pipeline (ties it together)
report.py               Markdown report generator (full trajectory + verdicts)
run_strategy.py         Single-run driver, emits METRIC lines for the loop
run_pipeline.py         Full-family pipeline driver (report + rejection log)
benchmark.sh            Entry point for the autonomous iteration loop
fetch_data.py           One-time data fetch (yfinance, cached to parquet)
tests/                  pytest invariants (no look-ahead, costs, metrics, stats)
data/                   Cached parquet, gitignored
results/                Per-run JSON + rejection log, gitignored
```

## Quick start

```bash
pip install yfinance pandas numpy scipy pyarrow pytest
python fetch_data.py

# Single strategy, full sample or walk-forward:
python run_strategy.py --strategy time_series_momentum
python run_strategy.py --strategy time_series_momentum --walk-forward

# Full multi-stage pipeline over all baselines (writes REPORT_v0.1.md
# and results/rejection_log.jsonl):
python run_pipeline.py

# Run the invariant tests (no network needed; uses synthetic data):
pytest tests/ -q
```

## Multi-stage verification pipeline

`run_pipeline.py` runs every pre-registered strategy through the same gauntlet:

1. **Walk-forward OOS** evaluation (24-month train / 1-month test, frozen
   parameters per fold, no overlap).
2. **Per-strategy pre-reg checks** against the locked success/failure
   thresholds in `PRE_REGISTRATION.md`.
3. **Holm-Bonferroni** multiple-comparison correction across the whole family
   at family-wise alpha 0.05.
4. **Capital-deployment gates** for any multiple-comparison survivor: 5x
   transaction cost (50 bps/side), S&P-100 liquidity-subset re-run, and a
   3-month train/test regime gap.
5. **Structured rejection logging**: every strategy, passed or rejected,
   records each check, its threshold, the realized metric, and the verdict to
   `results/rejection_log.jsonl`. Nothing is dropped silently.

## Autonomous iteration loop

`benchmark.sh` runs the active strategy and prints `METRIC name=value` lines on
stdout, so it can be driven by an autonomous iteration loop that proposes
strategies, runs the harness, and reads back the metrics. The search budget is
locked in the pre-reg: max 200 strategies, Holm-Bonferroni at family-wise alpha
0.05, and a hold-out re-run gate for any high-Sharpe candidate. A representative
search prompt:

> Search for strategies that beat the time_series_momentum baseline (OOS Sharpe
> 0.84) on the audited S&P large-cap harness. Hold the backtester, universe, TC,
> and slippage constant. Search over signal definition (momentum window 21 to
> 252 days, skip 0 to 21, top_n 5 to 50), additional features (volatility
> scaling, sector neutralization, beta hedge), and rebalance cadence (weekly to
> quarterly). Metric: Sharpe net of cost. Multiple-comparison gate:
> Holm-Bonferroni at family-wise alpha 0.05. Pre-reg gate: any candidate with
> Sharpe > 1.0 must be re-run on the 2024-2025 hold-out before being accepted.

## Pre-registration discipline

Every hypothesis is committed to `PRE_REGISTRATION.md` BEFORE any backtest is
run. Deviations are logged in `DEVIATION_LOG.md`, not silently fixed. The v0.1
to walk-forward move and the data-snapshot refresh are both recorded there,
including the before/after Sharpes. This is the same rule applied to the
Parchment Labs Sobol pilot (which violated its own consistency check and was
publicly logged) and the maze-rl-baselines harness audit (which caught a
survivorship-style filter and re-ran the affected experiments transparently).

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

MIT. See `LICENSE`.
