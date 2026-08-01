"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/operations/test_opal_precondition.py

Fail-closed OPAL readiness checks for the RT-lnRNA study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.ops.preflight import (
    ContractPreflightCheckDependencies,
    build_contract_preflight_checks,
    evaluate_preflight_checks,
)
from dnadesign.studies.core import build_study_preflight_plan, load_study_ops_contract

_EXPECTED_BLOCKERS = (
    "rt_lnrna.opal.selected_x_contract",
    "rt_lnrna.opal.label_projection_contract",
    "rt_lnrna.opal.comparable_profiles",
    "rt_lnrna.opal.objective_contract",
    "rt_lnrna.opal.training_table",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[7]


def test_opal_precondition_reports_each_intentionally_absent_machine_contract_as_blocking() -> None:
    repo_root = _repo_root()
    study_root = repo_root / "docs/studies/rt_lnrna_sponging_construct_triage"
    contract = load_study_ops_contract(study_root)
    checks = build_contract_preflight_checks(
        repo_root=repo_root,
        study_root=study_root,
        contract=contract,
        dataset_index={},
        execution_surface_index={},
        enabled_groups={"opal_precondition"},
        environ={},
        dependencies=ContractPreflightCheckDependencies(
            run_preflight_command=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("OPAL preconditions must not execute commands")
            ),
            safe_json_loads=lambda text: None,
            choose_command_summary=lambda *_args, fallback, **_kwargs: fallback,
            inspect_local_gpu_inventory=lambda: {"count": 0, "devices": [], "probe_error": None},
        ),
    )

    assert tuple(check.id for check in checks) == _EXPECTED_BLOCKERS
    assert all(check.kind == "path_exists" for check in checks)
    assert all(check.required and check.state == "missing" for check in checks)

    evaluation = evaluate_preflight_checks(
        checks,
        phase_states=[{"id": phase.id, "status": phase.status} for phase in contract.phases],
        scope_plan=build_study_preflight_plan(
            current_phase=contract.current_phase_id,
            next_ready_phase=None,
            scope="full",
            contract=contract.preflight,
        ),
    )
    assert evaluation.blocked_by_ids == tuple(sorted(_EXPECTED_BLOCKERS))
