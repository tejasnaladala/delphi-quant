"""v0.1 report generator for delphi-quant.

Emits a markdown report of the full search trajectory, including every failure
and rejection, the multiple-comparison-corrected significance, and the declared
protocol verdicts. The report is generated whether or not any candidate qualifies.

No cherry-picking: every strategy in the RejectionLog appears in the trajectory
table, in evaluation order, with its failed checks listed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rejection_log import RejectionLog
from stats import HolmResult


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def _verdict_emoji_free(v: str) -> str:
    # Plain text verdicts, no decoration.
    return v


def generate_report(
    log: RejectionLog,
    context: dict,
    title: str = "delphi-quant v0.1 verification report",
    evidence_kind: str = "unspecified",
    provenance: dict[str, object] | None = None,
) -> str:
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cfg = context.get("config", {})
    alpha = context.get("family_alpha", 0.05)

    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated {ts}.")
    lines.append("")
    if evidence_kind == "synthetic":
        lines.append(
            "> **SYNTHETIC LOGIC CHECK ONLY.** Every number below comes from a "
            "deterministic random panel. It is not market evidence, does not validate "
            "a strategy, and must not replace the canonical real-data report."
        )
        lines.append("")
    elif evidence_kind == "real":
        lines.append(
            "> **REAL-DATA COMPUTATION.** Reproducibility is tied to the input hash "
            "below. This code does not itself establish redistribution rights or data quality."
        )
        lines.append("")

    if provenance:
        lines.append("## Input provenance")
        lines.append("")
        for key, value in provenance.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    lines.append("## Evaluation setup")
    lines.append("")
    lines.append(f"- Transaction cost: {cfg.get('tc_bps', 'n/a')} bps per side")
    lines.append(f"- Slippage: {cfg.get('slippage_bps', 'n/a')} bps per side")
    lines.append(f"- Risk-free: {cfg.get('risk_free_annual', 'n/a')} annual")
    lines.append("- Evaluation: walk-forward OOS, 24-month train / 1-month test, frozen params per fold")
    lines.append(f"- Multiple-comparison correction: Holm-Bonferroni at family-wise alpha {alpha}")
    lines.append("")

    # Search trajectory: every strategy, including failures.
    lines.append("## Search trajectory (all strategies, including failures)")
    lines.append("")
    lines.append("| Strategy | OOS Sharpe | Raw p | Holm sig | Verdict | First failed check |")
    lines.append("|---|---|---|---|---|---|")
    for row in log.summary_rows():
        failed = log.first_failed_check(row["label"])
        failed_str = failed.check if failed else "-"
        holm_sig = "yes" if row["holm_significant"] else "no"
        lines.append(
            f"| {row['label']} | {_fmt(row['oos_sharpe'])} | {_fmt(row['raw_p_value'])} | "
            f"{holm_sig} | {_verdict_emoji_free(row['verdict'])} | {failed_str} |"
        )
    lines.append("")

    # Holm-Bonferroni table.
    holm: list[HolmResult] = context.get("holm", [])
    if holm:
        lines.append("## Holm-Bonferroni correction")
        lines.append("")
        lines.append(f"Family of {len(holm)} hypotheses, family-wise alpha {alpha}.")
        lines.append("")
        lines.append("| Rank | Strategy | Raw p | Adjusted alpha | Reject null |")
        lines.append("|---|---|---|---|---|")
        for h in holm:
            lines.append(
                f"| {h.rank} | {h.label} | {_fmt(h.p_value)} | "
                f"{_fmt(h.adjusted_alpha, 5)} | {'yes' if h.reject_null else 'no'} |"
            )
        lines.append("")

    # Per-strategy detail with every check.
    lines.append("## Per-strategy checks")
    lines.append("")
    for rec in log.records:
        lines.append(f"### {rec.label} - {rec.verdict}")
        lines.append("")
        lines.append(f"{rec.reason}")
        lines.append("")
        lines.append("| Check | Comparison | Threshold | Realized | Passed |")
        lines.append("|---|---|---|---|---|")
        for c in rec.checks:
            lines.append(
                f"| {c.check} | {c.comparison} | {_fmt(c.threshold, 4)} | "
                f"{_fmt(c.realized, 4)} | {'yes' if c.passed else 'no'} |"
            )
        lines.append("")

    # Deployment gates, if any candidate reached them.
    gates = context.get("gates", {})
    if gates:
        lines.append("## Capital-deployment gates")
        lines.append("")
        for name, glist in gates.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Gate | Realized Sharpe | Survival floor | Passed | Detail |")
            lines.append("|---|---|---|---|---|")
            for g in glist:
                lines.append(
                    f"| {g.name} | {_fmt(g.realized_sharpe)} | {_fmt(g.threshold)} | "
                    f"{'yes' if g.passed else 'no'} | {g.detail} |"
                )
            lines.append("")

    # Verdict summary.
    candidates = [r for r in log.records if r.verdict == "CANDIDATE"]
    lines.append("## Verdict")
    lines.append("")
    if evidence_kind == "synthetic":
        lines.append(
            "This run exercised the pipeline on synthetic data. Its labels and metrics are "
            "test outputs only; no candidate or deployment conclusion is supported."
        )
    elif candidates:
        names = ", ".join(r.label for r in candidates)
        lines.append(
            f"{len(candidates)} strategy/strategies cleared every stage and qualify as "
            f"v0.1 candidates: {names}. Per the declared protocol, a candidate is not "
            "deployment-ready; it still needs survivorship-bias-corrected data (v0.2) and "
            "several months of paper-trade OOS validation."
        )
    else:
        lines.append(
            "No strategy cleared every stage (declared target, Holm correction, and all "
            "deployment gates). This is an expected v0.1 outcome: the framework is built "
            "to reject multiple-comparison artifacts and cost-fragile strategies, not to "
            "manufacture a candidate."
        )
    lines.append("")
    return "\n".join(lines)
