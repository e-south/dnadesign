"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/status/test_preflight.py

Focused tests for the Retron hairpin design preflight service.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.ops.preflight import CommandExecution
from dnadesign.studies.units.retron_hairpin_design.status.preflight import (
    RetronHairpinDesignPreflightDependencies,
    build_retron_hairpin_design_preflight_progress,
    resolve_retron_hairpin_design_preflight_context,
)
from dnadesign.studies.units.retron_hairpin_design.status.service import STUDY_STATUS_SERVICE

from .test_snapshot import _write_retron_hairpin_design_record


def _execution(argv: tuple[str, ...], cwd: Path, *, returncode: int, stdout: str = "", stderr: str = "") -> object:
    return CommandExecution(
        argv=argv,
        cwd=str(cwd),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )


def test_retron_hairpin_design_preflight_next_scope_limits_checks_to_current_phase(tmp_path: Path) -> None:
    study_root = _write_retron_hairpin_design_record(tmp_path)
    context = STUDY_STATUS_SERVICE.load_context(repo_root=tmp_path, study_root=study_root)
    study_context = context.service_context.study_context

    resolved = resolve_retron_hairpin_design_preflight_context(
        study_context=study_context,
        scope="next",
        contract=context.contract,
    )

    state, summary, evidence = build_retron_hairpin_design_preflight_progress(
        context=resolved,
        dependencies=RetronHairpinDesignPreflightDependencies(
            run_preflight_command=lambda argv, *, cwd, timeout_seconds=180: _execution(  # noqa: ARG005
                tuple(argv),
                cwd,
                returncode=0,
            ),
            safe_json_loads=lambda text: {},
            choose_command_summary=lambda execution, *, fallback: fallback,
            environ={},
        ),
    )

    assert state == "ok"
    assert "blockers 0" in summary
    assert evidence["phase_id"] == "snapback_released_solve"
    assert evidence["included_groups"] == [
        "study_record",
        "snapback_workspace",
        "snapback_probe",
    ]
    scoped_ids = [check["id"] for check in evidence["checks"]]
    assert "de033.released_target_search" in scoped_ids
    assert "demo_monotypic_tetr.yiu_validate" not in scoped_ids


def test_retron_hairpin_design_preflight_reports_required_command_blockers(tmp_path: Path) -> None:
    study_root = _write_retron_hairpin_design_record(tmp_path)
    context = STUDY_STATUS_SERVICE.load_context(repo_root=tmp_path, study_root=study_root)
    study_context = context.service_context.study_context

    resolved = resolve_retron_hairpin_design_preflight_context(
        study_context=study_context,
        scope="next",
        contract=context.contract,
    )

    def _run_preflight_command(argv, *, cwd, timeout_seconds=180):  # noqa: ARG001
        argv_tuple = tuple(argv)
        if "released-target-search" in argv_tuple:
            return _execution(argv_tuple, cwd, returncode=1, stderr="probe failed")
        return _execution(argv_tuple, cwd, returncode=0)

    state, _, evidence = build_retron_hairpin_design_preflight_progress(
        context=resolved,
        dependencies=RetronHairpinDesignPreflightDependencies(
            run_preflight_command=_run_preflight_command,
            safe_json_loads=lambda text: {},
            choose_command_summary=lambda execution, *, fallback: fallback,
            environ={},
        ),
    )

    assert state == "attention"
    assert evidence["blocked_by_ids"] == ["de033.released_target_search"]
