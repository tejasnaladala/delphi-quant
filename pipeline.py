"""Multi-stage verification pipeline for delphi-quant.

Stages, in order:

  1. Walk-forward OOS evaluation of every strategy in the family (frozen params
     per fold, 24-month train / 1-month test, no overlap).
  2. Per-strategy pre-reg checks (the locked success/failure thresholds from
     PRE_REGISTRATION.md section 3).
  3. Holm-Bonferroni multiple-comparison correction across the whole family at
     family-wise alpha 0.05 (pre-reg section 4).
  4. Capital-deployment gates for any strategy that clears the pre-reg bar and
     survives the multiple-comparison correction (pre-reg section 4):
     5x TC, S&P-100 liquidity, 3-month regime gap.
  5. Structured rejection logging of every strategy (check, threshold,
     realized, verdict).

The verdict vocabulary:
  PASS         - sanity baseline behaved as pre-registered.
  ALIVE        - above the pre-reg "dead" threshold but not a deployment candidate.
  FAIL         - below the pre-reg dead threshold.
  REJECTED_MC  - Sharpe looked alive but not significant after Holm correction.
  CANDIDATE    - cleared pre-reg + Holm + all deployment gates.
  GATED        - cleared pre-reg + Holm but failed at least one deployment gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester import BacktestConfig
from gates import GateResult, run_all_gates
from rejection_log import CheckRecord, RejectionLog, StrategyRecord
from stats import holm_bonferroni, sharpe_pvalue
from strategies import STRATEGIES
from walk_forward import StrategyFn, run_walk_forward

FAMILY_ALPHA = 0.05

# Pre-registered per-strategy thresholds (PRE_REGISTRATION.md section 3).
PRE_REG = {
    "buy_and_hold": {
        "kind": "sanity",
        "sharpe_low": 0.4,
        "sharpe_high": 0.9,
        "dd_low": -0.45,
        "dd_high": -0.15,
    },
    "time_series_momentum": {
        "kind": "alpha",
        "target": 0.7,
        "dead": 0.5,
    },
    "cross_sectional_mean_reversion": {
        "kind": "alpha",
        "target": 0.5,
        "dead": 0.3,
    },
}

# Pre-reg deployment-gate trigger: a strategy must clear pre-reg + Holm before
# the deployment gates run. The pre-reg lists a Sharpe > 1.5 OOS hold-out gate
# for the autonomous search; for the three locked baselines we apply the gates
# to any strategy whose verdict is CANDIDATE-eligible (alpha kind, above target,
# Holm-significant) so the framework demonstrates the gate path end to end.


@dataclass(frozen=True)
class StrategyEval:
    label: str
    strategy: str
    params: dict
    oos: dict  # walk-forward OOS metrics
    raw_p: float
    checks: list[CheckRecord]


def _sanity_checks(name: str, oos: dict, spec: dict) -> tuple[list[CheckRecord], str, str]:
    sharpe = oos["sharpe"]
    dd = oos.get("max_dd", 0.0)
    sharpe_ok = spec["sharpe_low"] <= sharpe <= spec["sharpe_high"]
    dd_ok = spec["dd_low"] <= dd <= spec["dd_high"]
    checks = [
        CheckRecord(
            check="sanity_sharpe_in_range",
            threshold=spec["sharpe_low"],
            realized=sharpe,
            comparison=f"in [{spec['sharpe_low']}, {spec['sharpe_high']}]",
            passed=sharpe_ok,
            note=f"upper bound {spec['sharpe_high']}",
        ),
        CheckRecord(
            check="sanity_max_dd_in_range",
            threshold=spec["dd_high"],
            realized=dd,
            comparison=f"in [{spec['dd_low']}, {spec['dd_high']}]",
            passed=dd_ok,
            note=f"lower bound {spec['dd_low']}",
        ),
    ]
    if sharpe_ok and dd_ok:
        return checks, "PASS", "sanity baseline behaved within pre-registered ranges"
    return checks, "FAIL", "sanity baseline fell outside pre-registered ranges (possible backtester bug)"


def _alpha_checks(name: str, oos: dict, spec: dict) -> tuple[list[CheckRecord], str, str]:
    sharpe = oos["sharpe"]
    above_target = sharpe > spec["target"]
    above_dead = sharpe > spec["dead"]
    checks = [
        CheckRecord(
            check="oos_sharpe_above_target",
            threshold=spec["target"],
            realized=sharpe,
            comparison=">",
            passed=above_target,
        ),
        CheckRecord(
            check="oos_sharpe_above_dead_threshold",
            threshold=spec["dead"],
            realized=sharpe,
            comparison=">",
            passed=above_dead,
        ),
    ]
    if not above_dead:
        return checks, "FAIL", f"OOS Sharpe {sharpe:.3f} at or below pre-reg dead threshold {spec['dead']}"
    if not above_target:
        return checks, "ALIVE", f"OOS Sharpe {sharpe:.3f} above dead line but below pre-reg target {spec['target']}"
    return checks, "CANDIDATE_PENDING_MC", f"OOS Sharpe {sharpe:.3f} above pre-reg target {spec['target']}"


def evaluate_family(
    prices: pd.DataFrame,
    strategies: dict[str, StrategyFn] | None = None,
    cfg: BacktestConfig | None = None,
) -> tuple[RejectionLog, dict]:
    """Run the full multi-stage pipeline over a family of strategies.

    Returns the populated RejectionLog and a context dict with per-strategy OOS
    metrics, the Holm table, and per-candidate gate results.
    """
    cfg = cfg or BacktestConfig()
    strategies = strategies or STRATEGIES

    # Stage 1 + 2: walk-forward OOS + per-strategy pre-reg checks.
    evals: dict[str, StrategyEval] = {}
    for name, fn in strategies.items():
        oos = run_walk_forward(prices, fn, cfg=cfg)
        spec = PRE_REG.get(name, {"kind": "alpha", "target": 0.7, "dead": 0.5})
        if spec["kind"] == "sanity":
            checks, verdict0, reason0 = _sanity_checks(name, oos, spec)
        else:
            checks, verdict0, reason0 = _alpha_checks(name, oos, spec)
        raw_p = sharpe_pvalue(oos["sharpe"], oos["n_days"], cfg.trading_days_per_year)
        evals[name] = StrategyEval(
            label=name, strategy=name, params={}, oos=oos, raw_p=raw_p, checks=checks
        )
        evals[name].checks.append(
            CheckRecord(
                check="raw_sharpe_p_value",
                threshold=FAMILY_ALPHA,
                realized=raw_p,
                comparison="<=",
                passed=raw_p <= FAMILY_ALPHA,
                note="uncorrected two-sided p-value that OOS Sharpe != 0",
            )
        )
        # stash provisional verdict on the eval via params dict (kept simple)
        evals[name].params["_verdict0"] = verdict0
        evals[name].params["_reason0"] = reason0

    # Stage 3: Holm-Bonferroni across the family.
    pvals = {name: ev.raw_p for name, ev in evals.items()}
    holm = holm_bonferroni(pvals, family_alpha=FAMILY_ALPHA)
    holm_by_label = {h.label: h for h in holm}

    log = RejectionLog()
    gate_results: dict[str, list[GateResult]] = {}

    for name, ev in evals.items():
        h = holm_by_label[name]
        verdict0 = ev.params.pop("_verdict0")
        reason0 = ev.params.pop("_reason0")
        ev.checks.append(
            CheckRecord(
                check="holm_bonferroni_significant",
                threshold=h.adjusted_alpha,
                realized=ev.raw_p,
                comparison="<=",
                passed=h.reject_null,
                note=f"family alpha {FAMILY_ALPHA}, rank {h.rank}, adjusted alpha {h.adjusted_alpha:.5f}",
            )
        )

        verdict = verdict0
        reason = reason0

        # Stage 4: deployment gates for MC-surviving candidates.
        if verdict0 == "CANDIDATE_PENDING_MC":
            if not h.reject_null:
                verdict = "REJECTED_MC"
                reason = (
                    f"OOS Sharpe {ev.oos['sharpe']:.3f} above target but not significant "
                    f"after Holm correction (p={ev.raw_p:.3f} > adj alpha {h.adjusted_alpha:.5f})"
                )
            else:
                spec = PRE_REG[name]
                floor = spec["dead"]  # gate survival floor = strategy's own dead line
                gates = run_all_gates(prices, strategies[name], survival_floor=floor, cfg=cfg)
                gate_results[name] = gates
                for g in gates:
                    ev.checks.append(
                        CheckRecord(
                            check=f"gate_{g.name}",
                            threshold=g.threshold,
                            realized=g.realized_sharpe,
                            comparison=">=",
                            passed=g.passed,
                            note=g.detail,
                        )
                    )
                if all(g.passed for g in gates):
                    verdict = "CANDIDATE"
                    reason = "cleared pre-reg target, Holm correction, and all three deployment gates"
                else:
                    failed = [g.name for g in gates if not g.passed]
                    verdict = "GATED"
                    reason = f"cleared pre-reg + Holm but failed deployment gate(s): {', '.join(failed)}"

        log.add(
            StrategyRecord(
                label=ev.label,
                strategy=ev.strategy,
                params={k: v for k, v in ev.params.items() if not k.startswith("_")},
                checks=tuple(ev.checks),
                verdict=verdict,
                reason=reason,
                oos_sharpe=float(ev.oos["sharpe"]),
                raw_p_value=float(ev.raw_p),
                holm_adjusted_alpha=float(h.adjusted_alpha),
                holm_significant=bool(h.reject_null),
            )
        )

    context = {
        "oos": {name: ev.oos for name, ev in evals.items()},
        "holm": holm,
        "gates": gate_results,
        "family_alpha": FAMILY_ALPHA,
        "config": {
            "tc_bps": cfg.tc_bps,
            "slippage_bps": cfg.slippage_bps,
            "risk_free_annual": cfg.risk_free_annual,
        },
    }
    return log, context
