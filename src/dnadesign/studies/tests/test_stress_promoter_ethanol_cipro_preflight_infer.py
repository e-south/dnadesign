"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_stress_promoter_ethanol_cipro_preflight_infer.py

Focused tests for the study-owned infer and notify preflight builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.studies.stress_promoter_ethanol_cipro.infer_runtime import (
    PromoterStudyInferRuntimeModelSummary,
    PromoterStudyInferRuntimeResolvedContext,
)
from dnadesign.studies.stress_promoter_ethanol_cipro.preflight_infer import (
    PromoterPreflightInferDependencies,
    build_promoter_preflight_infer_checks,
)


def _state_check(**kwargs) -> dict[str, object]:
    return {
        "id": kwargs["check_id"],
        "phase": kwargs["phase"],
        "phase_id": kwargs["phase_id"],
        "state": kwargs["state"],
        "summary": kwargs["summary"],
        "details": kwargs.get("details", {}),
    }


def _command_check(**kwargs) -> dict[str, object]:
    execution = kwargs["execution"]
    return {
        "id": kwargs["check_id"],
        "phase": kwargs["phase"],
        "phase_id": kwargs["phase_id"],
        "state": kwargs.get("override_state", "ok"),
        "summary": kwargs["summary"],
        "details": kwargs.get("details", {}),
        "returncode": getattr(execution, "returncode", None),
    }


def test_build_promoter_preflight_infer_checks_reports_runtime_and_notify_contracts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.anchor_only.evo2_20b.yaml"
    full_lane_path = tmp_path / "config.full_lane_set.evo2_20b.yaml"
    profile_path = tmp_path / "outputs" / "notify" / "infer" / "anchor_only_20b" / "profile.json"
    events_path = tmp_path / "usr" / "promoter" / "demo" / ".events.log"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("event", encoding="utf-8")
    runtime_lane = SimpleNamespace(
        runtime_label="anchor_only_20b",
        config_path=config_path,
    )
    infer_runtime = PromoterStudyInferRuntimeResolvedContext(
        preferred_model_family="evo2_20b",
        supported_model_families=("evo2_20b", "evo2_7b"),
        infer_config_paths={
            "anchor_only_20b": config_path,
            "full_lane_set_20b": full_lane_path,
        },
        runtime_lane_contracts=(runtime_lane,),
        runtime_config_paths={"anchor_only_20b": config_path},
        phase_targets=(),
        phase_targets_by_id={},
        config_phase_ids={"anchor_only_20b": "infer_anchor_only_20b"},
        runtime_phase_ids={"anchor_only_20b": "infer_anchor_only_20b"},
        infer_notify_profile_paths={"anchor_only_20b": profile_path},
        infer_notify_profile_errors={},
        runtime_model_summaries=(
            PromoterStudyInferRuntimeModelSummary(
                label="anchor_only_20b",
                model_id="evo2_20b",
                device="cuda:0",
            ),
        ),
        gpu_required_runtime_labels=("anchor_only_20b",),
    )

    result = build_promoter_preflight_infer_checks(
        study_repo_root=tmp_path,
        infer_runtime=infer_runtime,
        infer_preparation_phase_id="infer_batch_preparation",
        include_infer_checks=True,
        include_notify_checks=True,
        dependencies=PromoterPreflightInferDependencies(
            inspect_local_gpu_inventory=lambda: {"count": 0, "devices": [], "probe_error": None},
            infer_usr_dataset_requirements=lambda _: [],
            build_infer_notify_setup_command=lambda path: f"setup:{path.name}",
            run_progress_command=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("notify profile doctor should not run for missing profile")
            ),
            preflight_state_check=_state_check,
            preflight_command_check=_command_check,
            choose_command_summary=lambda *_args, **_kwargs: "completed",
            validate_infer_config_contract=lambda _: SimpleNamespace(
                model_id="evo2_20b",
                device="cuda:0",
                job_ids=("job-1",),
                usr_datasets=("promoter/demo",),
            ),
            validate_infer_dry_run_contract=lambda _: SimpleNamespace(
                model_id="evo2_20b",
                device="cuda:0",
                job_ids=("job-1",),
            ),
            resolve_infer_usr_output_contract=lambda _: SimpleNamespace(
                usr_root=tmp_path / "usr",
                usr_dataset="promoter/demo",
            ),
        ),
    )

    checks = {check["id"]: check for check in result.checks}

    assert result.evidence_updates["preferred_infer_model_family"] == "evo2_20b"
    assert result.evidence_updates["supported_model_families"] == ["evo2_20b", "evo2_7b"]
    assert checks["infer.validate.anchor_only_20b"]["state"] == "ok"
    assert checks["infer.validate.full_lane_set_20b"]["state"] == "ok"
    assert checks["infer.local_runtime.anchor_only_20b"]["state"] == "attention"
    assert checks["notify.profile.anchor_only_20b"]["state"] == "attention"
    assert (
        checks["notify.profile.anchor_only_20b"]["details"]["setup_command"] == "setup:config.anchor_only.evo2_20b.yaml"
    )
    assert checks["infer.dry_run.anchor_only_20b"]["state"] == "ok"
    assert checks["notify.resolve_events.anchor_only_20b"]["state"] == "ok"


def test_build_promoter_preflight_infer_checks_requires_notify_resolver_dependency(tmp_path: Path) -> None:
    runtime_lane = SimpleNamespace(
        runtime_label="anchor_only_20b",
        config_path=tmp_path / "config.anchor_only.evo2_20b.yaml",
    )
    infer_runtime = PromoterStudyInferRuntimeResolvedContext(
        preferred_model_family=None,
        supported_model_families=(),
        infer_config_paths={},
        runtime_lane_contracts=(runtime_lane,),
        runtime_config_paths={"anchor_only_20b": runtime_lane.config_path},
        phase_targets=(),
        phase_targets_by_id={},
        config_phase_ids={},
        runtime_phase_ids={"anchor_only_20b": "infer_anchor_only_20b"},
        infer_notify_profile_paths={},
        infer_notify_profile_errors={},
        runtime_model_summaries=(
            PromoterStudyInferRuntimeModelSummary(
                label="anchor_only_20b",
                model_id="evo2_20b",
                device="cpu",
            ),
        ),
        gpu_required_runtime_labels=(),
    )

    with pytest.raises(ValueError, match="resolve_infer_usr_output_contract dependency"):
        build_promoter_preflight_infer_checks(
            study_repo_root=tmp_path,
            infer_runtime=infer_runtime,
            infer_preparation_phase_id="infer_batch_preparation",
            include_infer_checks=False,
            include_notify_checks=True,
            dependencies=PromoterPreflightInferDependencies(
                inspect_local_gpu_inventory=lambda: {"count": 0, "devices": [], "probe_error": None},
                infer_usr_dataset_requirements=lambda _: [],
                build_infer_notify_setup_command=lambda path: f"setup:{path.name}",
                run_progress_command=lambda *args, **kwargs: None,
                preflight_state_check=_state_check,
                preflight_command_check=_command_check,
                choose_command_summary=lambda *_args, **_kwargs: "completed",
            ),
        )
