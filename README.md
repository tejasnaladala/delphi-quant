# delphi-quant

`delphi-quant` is a Python research harness for three long-only equity
baselines. Each run follows one path from an explicit price snapshot to
out-of-sample metrics, statistical checks, stress gates, a Markdown report, and
a structured rejection log.

## Evidence status

This is a **retrospective, unverified research artifact**.

The first verifiable Git commit contains the protocol and implementation
together, so the repository does not establish prospective preregistration.
The price snapshot used for the June 2026 report is also absent and has no
recorded hash. `REPORT_v0.1.md` remains in the repository as an audit record;
its strategy metrics are not reproducible evidence.

Real-data runs require an explicit local parquet file and stop without writing
results when that file is missing. Synthetic runs require `--synthetic` and use
separate filenames by default. No result here supports capital deployment.

## Pipeline

```text
price parquet
  -> SHA-256 input manifest
  -> monthly out-of-sample evaluation
  -> retrospectively recorded strategy thresholds
  -> Sharpe significance tests + Holm-Bonferroni correction
  -> cost, liquidity, and train/test-gap stress gates
  -> Markdown report + JSONL rejection log
```

The implemented baseline family is:

- equal-weight buy and hold;
- long-only cross-sectional momentum using a 252-trading-day lookback, a
  21-day skip, monthly rebalancing, and the top 20 names;
- long-only five-day cross-sectional reversal, rebalanced weekly into the
  bottom 20 names.

The evaluator uses an initial 24 months of history, then scores consecutive
one-month windows while retaining the expanding history for signal warm-up.
Positions are lagged by one trading day. Default costs are 10 bps of transaction
cost and 5 bps of slippage per side.

Candidate checks include a fivefold transaction-cost and slippage stress, a
rerun on a fixed 50-ticker mega-cap list, and a scored window shifted three
months past the training cutoff. Signal warm-up remains continuous through that
gap. Every baseline receives a record with its thresholds, realized values,
pass/fail checks, p-value, Holm-Bonferroni decision, and final verdict.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Activate .venv with your shell, then:
python -m pip install yfinance pandas numpy scipy pyarrow pytest
python -m pytest -q
```

The tests use deterministic synthetic fixtures and make no network requests.

### Synthetic pipeline check

```bash
python run_pipeline.py --synthetic
```

This writes:

- `results/SYNTHETIC_REPORT.md`
- `results/synthetic_rejection_log.jsonl`
- `results/synthetic_input_manifest.json`

These outputs exercise the pipeline only. The default paths leave the
historical report untouched and carry no market-performance claim.

### Real-data run

```bash
python fetch_data.py
python run_pipeline.py \
  --data-path data/sp500_daily.parquet \
  --data-source-note "local yfinance snapshot; redistribution rights unverified"
```

`fetch_data.py` is the only step above that requires network access. A
successful pipeline run writes `results/input_manifest.json` with the exact
input hash, shape, and date range.

## Repository map

| Path | Contents |
|---|---|
| `pipeline.py` | Family evaluation, protocol checks, correction, and verdicts |
| `backtester.py` | Daily-bar accounting, lag, costs, and performance metrics |
| `walk_forward.py` | Monthly out-of-sample windows and aggregation |
| `strategies.py` | The three baseline strategy implementations |
| `stats.py` | Sharpe p-values and Holm-Bonferroni correction |
| `gates.py` | Cost, fixed-universe, and train/test-gap stress checks |
| `rejection_log.py` | Per-strategy JSONL records |
| `provenance.py` | Real and synthetic input manifests |
| `PRE_REGISTRATION.md` | Retrospective protocol record and evidence caveat |
| `DEVIATION_LOG.md` | Method changes and the August 2026 audit correction |
| `REPORT_v0.1.md` | Preserved, unverified historical report |

## Limitations

- The fetcher uses a hard-coded April 2026 large-cap universe. It is
  survivorship-biased and is not a point-in-time S&P 500 constituent history.
- Historical `S&P 500` and `S&P 100` labels are legacy names. The implemented
  universes are fixed large-cap lists of 104 and 50 tickers.
- The evaluator expands its history after the initial 24 months; it does not
  maintain a fixed 24-month rolling training window.
- Train and test slices share the cutoff timestamp because both ranges are
  inclusive. A fitted strategy would need a non-overlapping boundary before its
  score could be treated as clean out-of-sample evidence.
- `time_series_momentum` is the historical function name. Its implementation is
  cross-sectional ranking on lagged returns.
- The backtester enforces a one-day execution lag. It cannot detect future data
  read inside a strategy function; the tests document that failure mode.
- Daily bars and fixed basis-point costs omit intraday execution, spread
  dynamics, capacity, and market impact.
- The Sharpe p-value is a first-order normal approximation. It does not adjust
  for autocorrelation, non-normal returns, or broader data-snooping risk.
- The repository contains no immutable licensed market-data snapshot,
  point-in-time universe, prospective holdout, or paper-trading record.

MIT licensed. See `LICENSE`.
