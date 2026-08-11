"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_study_preflight_plan.py

Verify study preflight plan compilation and scope behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.ops.preflight import build_state_check, evaluate_preflight_checks
from dnadesign.ops.study import (
    StudyPreflightContract,
    build_study_preflight_plan,
    compile_study_preflight_execution_plan,
)


def _contract() -> StudyPreflightContract:
    return StudyPreflightContract(
        default_scope="next",
        scope_groups={
            "next": ("study_record", "compiler"),
            "full": ("study_record", "compiler", "optional_review"),
        },
    )


def test_build_study_preflight_plan_uses_explicit_groups() -> None:
    plan = build_study_preflight_plan(scope="next", contract=_contract())

    assert plan.scope == "next"
    assert plan.included_groups == ("study_record", "compiler")


def test_build_study_preflight_plan_full_scope_uses_all_declared_groups() -> None:
    plan = build_study_preflight_plan(scope="full", contract=_contract())

    assert plan.scope == "full"
    assert plan.included_groups == ("study_record", "compiler", "optional_review")


def test_compile_study_preflight_checks_uses_check_set_and_derived_category() -> None:
    contract = StudyPreflightContract(
        default_scope="next",
        scope_groups={"next": ("compiler_plan",), "full": ("compiler_plan",)},
        check_specs={
            "compiler_readiness": (
                {
                    "kind": "runbook_plan",
                    "check_id": "compiler.plan",
                    "check_group": "compiler_plan",
                    "summary": "Compiler plan is valid.",
                    "required": True,
                    "surface": "compiler_runbook",
                },
            )
        },
    )

    execution_plan = compile_study_preflight_execution_plan(
        contract=contract,
        enabled_groups=("compiler_plan",),
    )

    assert len(execution_plan.checks) == 1
    check = execution_plan.checks[0]
    assert check.check_set_id == "compiler_readiness"
    assert check.category == "ops"
    assert check.payload == {"surface": "compiler_runbook"}


def test_evaluate_checks_filters_by_scope_and_defers_other_required_failures() -> None:
    checks = [
        build_state_check(
            check_id="record.present",
            check_group="study_record",
            category="record",
            check_set_id="study_record",
            state="ok",
            summary="record ok",
        ),
        build_state_check(
            check_id="compiler.ready",
            check_group="compiler",
            category="compiler",
            check_set_id="compiler",
            state="attention",
            summary="compiler attention",
        ),
        build_state_check(
            check_id="review.optional",
            check_group="optional_review",
            category="review",
            check_set_id="optional_review",
            state="missing",
            summary="review missing",
        ),
    ]

    evaluation = evaluate_preflight_checks(
        checks,
        scope_plan=build_study_preflight_plan(scope="next", contract=_contract()),
    )

    assert evaluation.blocked_by_ids == ("compiler.ready",)
    assert evaluation.deferred_check_ids == ("review.optional",)
    assert evaluation.scoped_counts == {"ok": 1, "attention": 1, "missing": 0}


def test_optional_attention_is_advisory() -> None:
    check = build_state_check(
        check_id="review.note",
        required=False,
        check_group="compiler",
        category="review",
        check_set_id="optional_review",
        state="attention",
        summary="review note",
    )

    evaluation = evaluate_preflight_checks(
        [check],
        scope_plan=build_study_preflight_plan(scope="next", contract=_contract()),
    )

    assert evaluation.blocked_by_ids == ()
    assert evaluation.nonblocking_attention_ids == ("review.note",)
