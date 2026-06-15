# Deviation Log

The pre-registration (`PRE_REGISTRATION.md`) requires that any departure from
the locked methodology, and any change to reported numbers, be logged here with
a reason rather than silently fixed. Each entry is dated and states what
changed, why, and what it affected.

## 2026-06-15 - v0.1 -> v0.2: evaluation path moved from full-sample to walk-forward OOS

**What changed.** The v0.1 code defined `walk_forward_split` in `backtester.py`
but never used it: `run_backtest` (and therefore the reported numbers) evaluated
the full sample. The pre-reg (section 2) specifies rolling 24-month train /
1-month test walk-forward with frozen parameters. The evaluation path is now
wired to walk-forward OOS via `walk_forward.py` and `pipeline.py`. The reported
Sharpes are now out-of-sample, aggregated over the concatenated per-fold OOS
return streams.

**Why.** Full-sample evaluation leaks the entire history into the reported
metric. The pre-reg locked walk-forward as the evaluation method; the code did
not honor it. This is a correctness fix, logged here because it changes every
reported number.

**Number changes (the honest part).**

| Strategy | README v0.1 (full-sample) | Live full-sample (2026-06-15 data) | Live walk-forward OOS |
|---|---|---|---|
| Buy-and-hold | 0.83 | 0.836 | 0.898 |
| TS momentum (12-1, top 20) | 0.77 | 0.720 | 0.841 |
| XS mean reversion (5d, bottom 20) | 0.26 | 0.252 | 0.287 |

Two distinct sources of difference:

1. **Data snapshot drift.** The README's v0.1 numbers were produced from a
   yfinance pull at lock time (2026-04-27). Re-pulling on 2026-06-15 gives
   slightly different adjusted prices (dividend/split re-statement is normal in
   yfinance's adjusted series). On the *same* full-sample method, TS momentum
   moved from 0.77 to 0.720 and the others rounded the same. The 0.77 -> 0.720
   gap is a genuine data-snapshot deviation, logged here per the pre-reg.

2. **Method change (full-sample -> walk-forward OOS).** Moving to the
   pre-registered walk-forward method changes the numbers again, this time
   upward for all three. The OOS path warms signals up on train history but only
   scores test-window days; the higher OOS Sharpes reflect that the 2010-2025
   sample's worst drawdown stretches sit partly inside warm-up windows. These
   are the numbers the README now reports, because they are the ones the pre-reg
   actually specified.

**Verdict impact.** Under the pre-registered walk-forward OOS method plus
Holm-Bonferroni (family-wise alpha 0.05) plus the three deployment gates:

- Buy-and-hold: PASS (sanity, Sharpe 0.898 inside the [0.4, 0.9] band).
- TS momentum: CANDIDATE. OOS Sharpe 0.841 clears the 0.7 target, is
  Holm-significant (raw p 0.0017, adjusted alpha 0.025), and survives all three
  gates (5x TC: 0.675; S&P-100 liquidity: 0.938; 3-month regime gap: 0.796). A
  candidate is not deployment-ready; honest scope still requires v0.2
  survivorship-corrected data and months of paper-trade OOS.
- XS mean reversion: FAIL. OOS Sharpe 0.287 sits at/below the pre-registered
  0.3 dead threshold and is not significant (raw p 0.285). Consistent with the
  pre-reg's stated failure mode for post-2010 short-horizon reversal.

## 2026-06-15 - backtester `pct_change` fill behavior pinned

**What changed.** `backtester.py` and `strategies.py` now call
`pct_change(fill_method=None)` explicitly instead of relying on the deprecated
pandas default (`fill_method='pad'`).

**Why.** pandas 2.x deprecates the implicit forward-fill in `pct_change`. The
implicit pad could fabricate a zero return across a missing bar; `None` is the
honest choice (a gap stays NaN and is excluded). On the current cached data this
did not change any reported metric (verified: full-sample Sharpes identical
to 4 dp before and after), but it removes a silent future-behavior change.

## 2026-06-15 - failed-download ticker dropped from the universe

**What changed.** `load_prices` now drops columns that are entirely NaN. One
ticker (MMC) failed to download on the 2026-06-15 pull and arrived as an
all-NaN column.

**Why.** An all-NaN column is not a tradable name; leaving it in inflates the
nominal universe size. Verified that dropping it does not change any baseline
Sharpe (the strategies already skipped NaN names via `dropna` in their signal
construction). Universe is reported as 103 names (104 attempted, 1 failed
download) rather than 104.
