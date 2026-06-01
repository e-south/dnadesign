"""Execution contracts for DenseGen probe sweep planning."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from ..core.artifacts import ProbePlan
from ..core.constants import RUN_STAGES

_MANY_CAMPAIGN_THRESHOLD = 12


def build_sweep_execution_contract(plan: ProbePlan) -> dict[str, Any]:
    command_counts = _opal_command_counts(plan.commands)
    scored = _is_scored_plan(plan)
    planned_runs = len(plan.runs)
    expected_round_rows = planned_runs * int(plan.rounds) if scored else 0
    expected_run_rows = planned_runs if scored else 0
    expected_final_labeled = (
        int(plan.initial_label_count) + max(int(plan.rounds) - 1, 0) * int(plan.selection_k) if scored else None
    )
    blocking = _blocking_problems(plan)
    warnings = _warnings(plan)

    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.sweep_execution_contract.v1",
        "status": "blocked" if blocking else ("attention" if warnings else "ok"),
        "blocking_problems": blocking,
        "warnings": warnings,
        "planned_campaign_count": planned_runs,
        "planned_opal_command_count": len(plan.commands),
        "opal_command_counts": command_counts,
        "planned_round_count": expected_round_rows,
        "expected_run_metric_rows": expected_run_rows,
        "expected_round_metric_rows": expected_round_rows,
        "expected_selection_artifacts": expected_round_rows,
        "expected_final_round": int(plan.rounds) - 1 if scored else None,
        "expected_final_labeled_ids_per_campaign": expected_final_labeled,
        "expected_selection_ids_per_campaign": int(plan.rounds) * int(plan.selection_k) if scored else 0,
        "single_seed": int(plan.seed),
        "suite_seeds": list(plan.suite_seeds),
        "suite_seed_count": len(plan.suite_seeds),
        "suite_campaign_count_if_repeated_for_all_suite_seeds": planned_runs * len(plan.suite_seeds),
        "score_batch_size": None if plan.score_batch_size is None else int(plan.score_batch_size),
    }


def enforce_sweep_apply_contract(plan: ProbePlan) -> None:
    problems = _blocking_problems(plan)
    if problems:
        joined = "; ".join(problems)
        raise ValueError(f"DenseGen probe sweep apply contract failed: {joined}")


def _blocking_problems(plan: ProbePlan) -> list[str]:
    if not _is_scored_plan(plan):
        return []
    if len(plan.runs) < _MANY_CAMPAIGN_THRESHOLD:
        return []
    if plan.score_batch_size is None:
        return ["score_batch_size_required_for_many_campaign_scored_apply"]
    return []


def _warnings(plan: ProbePlan) -> list[str]:
    warnings: list[str] = []
    if _is_scored_plan(plan) and str(plan.gate or "").strip().lower() != "all":
        warnings.append("scored_run_is_not_full_all_gate")
    return warnings


def _is_scored_plan(plan: ProbePlan) -> bool:
    if not plan.runs:
        return False
    return RUN_STAGES.index(plan.stop_after) >= RUN_STAGES.index("run")


def _opal_command_counts(commands: Sequence[Sequence[str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for command in commands:
        if len(command) > 3 and list(command[:3]) == ["uv", "run", "opal"]:
            counts[str(command[3])] += 1
    return dict(sorted(counts.items()))
