# delphi-quant v0.1 verification report

Generated 2026-06-15 11:45 UTC.

## Evaluation setup

- Transaction cost: 10.0 bps per side
- Slippage: 5.0 bps per side
- Risk-free: 0.04 annual
- Evaluation: walk-forward OOS, 24-month train / 1-month test, frozen params per fold
- Multiple-comparison correction: Holm-Bonferroni at family-wise alpha 0.05

## Search trajectory (all strategies, including failures)

| Strategy | OOS Sharpe | Raw p | Holm sig | Verdict | First failed check |
|---|---|---|---|---|---|
| buy_and_hold | 0.898 | 0.001 | yes | PASS | - |
| time_series_momentum | 0.841 | 0.002 | yes | CANDIDATE | - |
| cross_sectional_mean_reversion | 0.287 | 0.285 | no | FAIL | oos_sharpe_above_target |

## Holm-Bonferroni correction

Family of 3 hypotheses, family-wise alpha 0.05.

| Rank | Strategy | Raw p | Adjusted alpha | Reject null |
|---|---|---|---|---|
| 1 | buy_and_hold | 0.001 | 0.01667 | yes |
| 2 | time_series_momentum | 0.002 | 0.02500 | yes |
| 3 | cross_sectional_mean_reversion | 0.285 | 0.05000 | no |

## Per-strategy checks

### buy_and_hold - PASS

sanity baseline behaved within pre-registered ranges

| Check | Comparison | Threshold | Realized | Passed |
|---|---|---|---|---|
| sanity_sharpe_in_range | in [0.4, 0.9] | 0.4000 | 0.8981 | yes |
| sanity_max_dd_in_range | in [-0.45, -0.15] | -0.1500 | -0.3348 | yes |
| raw_sharpe_p_value | <= | 0.0500 | 0.0008 | yes |
| holm_bonferroni_significant | <= | 0.0167 | 0.0008 | yes |

### time_series_momentum - CANDIDATE

cleared pre-reg target, Holm correction, and all three deployment gates

| Check | Comparison | Threshold | Realized | Passed |
|---|---|---|---|---|
| oos_sharpe_above_target | > | 0.7000 | 0.8413 | yes |
| oos_sharpe_above_dead_threshold | > | 0.5000 | 0.8413 | yes |
| raw_sharpe_p_value | <= | 0.0500 | 0.0017 | yes |
| holm_bonferroni_significant | <= | 0.0250 | 0.0017 | yes |
| gate_tc_5x_sensitivity | >= | 0.5000 | 0.6746 | yes |
| gate_sp100_liquidity | >= | 0.5000 | 0.9382 | yes |
| gate_regime_gap_3mo | >= | 0.5000 | 0.7962 | yes |

### cross_sectional_mean_reversion - FAIL

OOS Sharpe 0.287 at or below pre-reg dead threshold 0.3

| Check | Comparison | Threshold | Realized | Passed |
|---|---|---|---|---|
| oos_sharpe_above_target | > | 0.5000 | 0.2870 | no |
| oos_sharpe_above_dead_threshold | > | 0.3000 | 0.2870 | no |
| raw_sharpe_p_value | <= | 0.0500 | 0.2847 | no |
| holm_bonferroni_significant | <= | 0.0500 | 0.2847 | no |

## Capital-deployment gates

### time_series_momentum

| Gate | Realized Sharpe | Survival floor | Passed | Detail |
|---|---|---|---|---|
| tc_5x_sensitivity | 0.675 | 0.500 | yes | OOS Sharpe at 50 bps/side TC + 25 bps slippage |
| sp100_liquidity | 0.938 | 0.500 | yes | OOS Sharpe on 50-name liquid subset |
| regime_gap_3mo | 0.796 | 0.500 | yes | OOS Sharpe with 3-month train/test gap (164 folds) |

## Verdict

1 strategy/strategies cleared every stage and qualify as v0.1 candidates: time_series_momentum. Per the pre-reg honest scope, a candidate is not deployment-ready; it still needs survivorship-bias-corrected data (v0.2) and several months of paper-trade OOS validation.
