"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_stress_promoter_ethanol_cipro_preflight_scope.py

Focused tests for the study-owned preflight scope planner and blocker sorting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.ops.contracts import InferRuntimePhaseTarget
from dnadesign.studies.stress_promoter_ethanol_cipro.preflight_scope import (
    build_promoter_preflight_scope_plan,
    evaluate_promoter_preflight_checks,
)


def _target(
    *, phase_id: str, runtime_label: str, config_label: str, runbook_surface_label: str
) -> InferRuntimePhaseTarget:
    return InferRuntimePhaseTarget(
        phase_id=phase_id,
        runtime_label=runtime_label,
        config_label=config_label,
        runbook_surface_label=runbook_surface_label,
    )


def test_build_promoter_preflight_scope_plan_limits_groups_for_lane_scope() -> None:
    infer_phase_targets = {
        "infer_anchor_only_20b": _target(
            phase_id="infer_anchor_only_20b",
            runtime_label="anchor_only_20b",
            config_label="anchor_only_20b",
            runbook_surface_label="infer_batch_20b_with_notify.anchor_only",
        )
    }

    plan = build_promoter_preflight_scope_plan(
        current_phase="infer_anchor_only_20b",
        next_ready_phase=None,
        scope="next",
        infer_phase_targets=infer_phase_targets,
    )

    assert plan.scope == "next"
    assert plan.target_phase_id == "infer_anchor_only_20b"
    assert not plan.include_densegen_checks
    assert not plan.include_construct_checks
    assert plan.include_infer_checks
    assert plan.include_notify_checks
    assert plan.include_infer_batch_plan_checks


def test_evaluate_promoter_preflight_checks_demotes_completed_phase_attention_in_full_scope() -> None:
    checks = [
        {"id": "infer.local_runtime.anchor_only_20b", "state": "attention", "phase_id": "infer_anchor_only_20b"},
        {"id": "notify.profile.anchor_only_20b", "state": "attention", "phase_id": "infer_anchor_only_20b"},
        {
            "id": "infer.local_runtime.anchor_plus_template_20b",
            "state": "attention",
            "phase_id": "infer_anchor_plus_template_20b",
        },
    ]
    phase_states = [
        {"id": "infer_anchor_only_20b", "status": "complete"},
        {"id": "infer_anchor_plus_template_20b", "status": "planned"},
    ]
    plan = build_promoter_preflight_scope_plan(
        current_phase="infer_batch_preparation",
        next_ready_phase=None,
        scope="full",
        infer_phase_targets={},
    )

    evaluation = evaluate_promoter_preflight_checks(
        checks,
        phase_states=phase_states,
        scope_plan=plan,
        infer_phase_targets={},
    )

    assert evaluation.blocked_by_ids == ("infer.local_runtime.anchor_plus_template_20b",)
    assert evaluation.nonblocking_attention_ids == (
        "infer.local_runtime.anchor_only_20b",
        "notify.profile.anchor_only_20b",
    )


def test_evaluate_promoter_preflight_checks_defers_downstream_lane_blockers_in_next_scope() -> None:
    infer_phase_targets = {
        "infer_anchor_only_20b": _target(
            phase_id="infer_anchor_only_20b",
            runtime_label="anchor_only_20b",
            config_label="anchor_only_20b",
            runbook_surface_label="infer_batch_20b_with_notify.anchor_only",
        ),
        "infer_anchor_only_7b": _target(
            phase_id="infer_anchor_only_7b",
            runtime_label="anchor_only_7b",
            config_label="anchor_only_7b",
            runbook_surface_label="infer_batch_7b_with_notify.anchor_only",
        ),
    }
    checks = [
        {"id": "notify.environment.webhook", "state": "attention", "phase_id": "infer_batch_preparation"},
        {"id": "infer.local_runtime.anchor_only_20b", "state": "attention", "phase_id": "infer_anchor_only_20b"},
        {
            "id": "ops.runbook_plan.infer_batch_20b_with_notify.anchor_only",
            "state": "attention",
            "phase_id": "infer_anchor_only_20b",
        },
        {"id": "infer.local_runtime.anchor_only_7b", "state": "attention", "phase_id": "infer_anchor_only_7b"},
    ]
    plan = build_promoter_preflight_scope_plan(
        current_phase="infer_anchor_only_20b",
        next_ready_phase=None,
        scope="next",
        infer_phase_targets=infer_phase_targets,
    )

    evaluation = evaluate_promoter_preflight_checks(
        checks,
        phase_states=[
            {"id": "infer_anchor_only_20b", "status": "planned"},
            {"id": "infer_anchor_only_7b", "status": "planned"},
        ],
        scope_plan=plan,
        infer_phase_targets=infer_phase_targets,
    )

    assert evaluation.blocked_by_ids == (
        "infer.local_runtime.anchor_only_20b",
        "notify.environment.webhook",
        "ops.runbook_plan.infer_batch_20b_with_notify.anchor_only",
    )
    assert evaluation.deferred_check_ids == ("infer.local_runtime.anchor_only_7b",)
