"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_study_preflight_plan.py

Focused tests for the generic study-preflight scope planner and blocker sorting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.core.models import StudyPreflightContract, StudyPreflightNextScopeContract
from dnadesign.studies.core.preflight_plan import build_study_preflight_plan, evaluate_study_preflight_checks


def _contract() -> StudyPreflightContract:
    return StudyPreflightContract(
        default_scope="next",
        group_phase_bindings={
            "densegen": "densegen_growth",
            "construct": "construct_context_expansion",
            "notify_environment": "infer_batch_preparation",
        },
        next_scope=StudyPreflightNextScopeContract(
            target_phase_groups={
                "densegen_growth": ("densegen",),
                "construct_context_expansion": ("construct",),
                "infer_batch_preparation": ("infer", "notify_environment", "notify", "infer_batch_plan"),
            },
            runtime_phase_groups=("infer", "notify", "infer_batch_plan"),
            runtime_shared_groups=("notify_environment",),
        ),
    )


def test_build_study_preflight_plan_limits_runtime_groups_for_lane_scope() -> None:
    plan = build_study_preflight_plan(
        current_phase="infer_anchor_only_20b",
        next_ready_phase=None,
        scope="next",
        contract=_contract(),
        runtime_phase_ids=("infer_anchor_only_20b",),
    )

    assert plan.scope == "next"
    assert plan.target_phase_id == "infer_anchor_only_20b"
    assert plan.included_groups == ("notify_environment", "infer", "notify", "infer_batch_plan")
    assert plan.phase_scoped_groups == ("infer", "notify", "infer_batch_plan")


def test_evaluate_study_preflight_checks_demotes_completed_phase_attention_in_full_scope() -> None:
    checks = [
        {
            "id": "infer.local_runtime.anchor_only_20b",
            "check_group": "infer",
            "state": "attention",
            "phase_id": "infer_anchor_only_20b",
        },
        {
            "id": "notify.profile.anchor_only_20b",
            "check_group": "notify",
            "state": "attention",
            "phase_id": "infer_anchor_only_20b",
        },
        {
            "id": "infer.local_runtime.anchor_plus_template_20b",
            "check_group": "infer",
            "state": "attention",
            "phase_id": "infer_anchor_plus_template_20b",
        },
    ]
    evaluation = evaluate_study_preflight_checks(
        checks,
        phase_states=[
            {"id": "infer_anchor_only_20b", "status": "complete"},
            {"id": "infer_anchor_plus_template_20b", "status": "planned"},
        ],
        scope_plan=build_study_preflight_plan(
            current_phase="infer_batch_preparation",
            next_ready_phase=None,
            scope="full",
            contract=_contract(),
            runtime_phase_ids=(),
        ),
    )

    assert evaluation.blocked_by_ids == ("infer.local_runtime.anchor_plus_template_20b",)
    assert evaluation.nonblocking_attention_ids == (
        "infer.local_runtime.anchor_only_20b",
        "notify.profile.anchor_only_20b",
    )


def test_evaluate_study_preflight_checks_defers_downstream_lane_blockers_in_next_scope() -> None:
    checks = [
        {
            "id": "notify.environment.webhook",
            "check_group": "notify_environment",
            "state": "attention",
            "phase_id": "infer_batch_preparation",
        },
        {
            "id": "infer.local_runtime.anchor_only_20b",
            "check_group": "infer",
            "state": "attention",
            "phase_id": "infer_anchor_only_20b",
        },
        {
            "id": "ops.runbook_plan.infer_batch_20b_with_notify.anchor_only",
            "check_group": "infer_batch_plan",
            "state": "attention",
            "phase_id": "infer_anchor_only_20b",
        },
        {
            "id": "infer.local_runtime.anchor_only_7b",
            "check_group": "infer",
            "state": "attention",
            "phase_id": "infer_anchor_only_7b",
        },
    ]
    evaluation = evaluate_study_preflight_checks(
        checks,
        phase_states=[
            {"id": "infer_anchor_only_20b", "status": "planned"},
            {"id": "infer_anchor_only_7b", "status": "planned"},
        ],
        scope_plan=build_study_preflight_plan(
            current_phase="infer_anchor_only_20b",
            next_ready_phase=None,
            scope="next",
            contract=_contract(),
            runtime_phase_ids=("infer_anchor_only_20b", "infer_anchor_only_7b"),
        ),
    )

    assert evaluation.blocked_by_ids == (
        "infer.local_runtime.anchor_only_20b",
        "notify.environment.webhook",
        "ops.runbook_plan.infer_batch_20b_with_notify.anchor_only",
    )
    assert evaluation.deferred_check_ids == ("infer.local_runtime.anchor_only_7b",)
