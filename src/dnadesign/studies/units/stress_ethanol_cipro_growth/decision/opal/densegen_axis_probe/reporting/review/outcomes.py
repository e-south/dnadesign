"""Outcome summaries for DenseGen axis probe review artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def outcome_summary_payload(
    *,
    decision: str,
    status: str,
    review_problems: Sequence[str],
    decision_reasons: Sequence[Mapping[str, Any]],
    gate_coverage: Mapping[str, Any],
    trajectory_qa: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a compact, non-overclaiming explanation of the probe outcome."""

    problem_count = len(review_problems)
    reason_count = len(decision_reasons)
    pair_count = len(trajectory_qa.get("pairs") or []) if isinstance(trajectory_qa, Mapping) else 0
    coverage_text = _coverage_text(gate_coverage)
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.outcome_summary.v1",
        "headline": _headline(decision),
        "decision": decision,
        "status": status,
        "coverage": coverage_text,
        "problem_count": problem_count,
        "decision_reason_count": reason_count,
        "trajectory_pair_count": pair_count,
        "interpretation_boundary": (
            "This is a pre-assay synthetic-oracle learnability probe. It does not claim measured growth, stress "
            "tolerance, TF binding, regulatory mechanism, wet-lab phenotype, or biological causality."
        ),
        "operator_read": _operator_read(
            decision=decision,
            status=status,
            problem_count=problem_count,
            reason_count=reason_count,
            pair_count=pair_count,
            coverage_text=coverage_text,
        ),
        "next_action": _next_action(decision),
    }


def _headline(decision: str) -> str:
    if decision.startswith("PASS_"):
        return "Configured synthetic-oracle gate passed for the scored coverage."
    if decision == "DEBUG":
        return "Scored evidence is present, but at least one gate needs debugging before expansion."
    if decision == "STOP":
        return "A hard safety or artifact contract failed; stop before extending the probe."
    if decision == "PENDING":
        return "The run root is not scored yet, so no learnability outcome is available."
    return "The probe produced an unrecognized decision token."


def _operator_read(
    *,
    decision: str,
    status: str,
    problem_count: int,
    reason_count: int,
    pair_count: int,
    coverage_text: str,
) -> str:
    if decision.startswith("PASS_"):
        return (
            f"The probe passed within {coverage_text}. Review {pair_count} paired trajectory rows before treating the "
            "result as repeatable campaign evidence."
        )
    if decision == "DEBUG":
        return (
            f"The review status is {status} with {problem_count} contract problem(s) and {reason_count} decision "
            "reason(s). Inspect positive/null separation and round dynamics before adding seeds or label families."
        )
    if decision == "STOP":
        return (
            f"The review status is {status} with {problem_count} contract problem(s). Fix the failed contract before "
            "running or reporting additional probe campaigns."
        )
    if decision == "PENDING":
        return "Materialization may be present, but scored OPAL run metrics are absent."
    return "Treat this result as invalid until the decision-token contract is repaired."


def _next_action(decision: str) -> str:
    if decision.startswith("PASS_"):
        return "Expand only inside the declared coverage, or add a new configured campaign set with explicit gates."
    if decision == "DEBUG":
        return "Debug the weakest gate, refresh configured plots, and rerun the report before expansion."
    if decision == "STOP":
        return "Repair the hard contract failure, then rerun status and report generation."
    if decision == "PENDING":
        return "Run the scoped OPAL campaign gate, then regenerate metrics and review artifacts."
    return "Repair the decision contract."


def _coverage_text(gate_coverage: Mapping[str, Any]) -> str:
    campaigns = ", ".join(gate_coverage.get("campaigns") or []) or "no campaigns"
    families = ", ".join(gate_coverage.get("label_families") or []) or "no label families"
    splits = ", ".join(gate_coverage.get("splits") or []) or "no splits"
    return f"campaigns [{campaigns}], label families [{families}], splits [{splits}]"
