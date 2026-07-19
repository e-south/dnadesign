"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/operations/status/test_snapshot.py

Focused tests for the study-owned snapshot coordination layer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.studies.core.models import StudyOpsContract, StudyPreflightContract, StudyStatusContext
from dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.downstream_surfaces import (
    inspect_stress_ethanol_cipro_growth_downstream_surfaces,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.infer_runtime import (
    StressEthanolCiproGrowthInferRuntimeDependencies,
    StressEthanolCiproGrowthInferRuntimeResolvedContext,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.record_normalizer import (
    StressEthanolCiproGrowthResolvedContext,
    _first_phase_by_status,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service import (
    STUDY_STATUS_SERVICE,
    StressEthanolCiproGrowthStatusServiceContext,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.snapshot import (
    StressEthanolCiproGrowthStatusDependencies,
    StressEthanolCiproGrowthStatusResolvedContext,
    _build_planned_outputs_state,
    build_stress_ethanol_cipro_growth_status,
    resolve_stress_ethanol_cipro_growth_status_context,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list_or_empty(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _string_or_none(item)
        if text is not None:
            result.append(text)
    return result


def test_stress_ethanol_cipro_status_service_source_is_decomposed_by_responsibility() -> None:
    source_root = (
        _repo_root()
        / "src"
        / "dnadesign"
        / "studies"
        / "units"
        / "stress_ethanol_cipro_growth"
        / "operations"
        / "status"
    )
    budgets = {
        "service.py": 320,
        "probes/runtime_dependencies.py": 140,
        "probes/semantic_completeness.py": 200,
        "probes/sequence_view_contracts.py": 240,
    }

    for relative_path, max_lines in budgets.items():
        path = source_root / relative_path
        assert path.is_file(), relative_path
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert line_count <= max_lines, f"{relative_path} has {line_count} lines > {max_lines}"


def test_stress_ethanol_cipro_next_planned_phase_skips_nonblocking_reference_branch() -> None:
    phases = [
        {"id": "genbank_reference_import", "status": "planned", "required_for_main_study_state": False},
        {"id": "infer_anchor_only_20b", "status": "planned", "required_for_main_study_state": True},
    ]

    assert _first_phase_by_status(phases, status="planned", require_main_study_state=True) == phases[1]
    assert _first_phase_by_status(phases, status="planned") == phases[0]


def test_planned_outputs_ignore_completed_logical_output_datasets(tmp_path: Path) -> None:
    study_context = _make_study_context(tmp_path)
    study_context = replace(
        study_context,
        phase_states=(
            *study_context.phase_states,
            {
                "id": "feature_sidecar_export",
                "status": "complete",
                "output_dataset": "usr_demo_opal_candidates",
            },
        ),
    )

    assert _build_planned_outputs_state(study_context=study_context) == {
        "state": "ok",
        "pending_datasets": [],
        "drives_top_level_attention": False,
        "include_in_summary": False,
        "summary": "planned outputs clear",
    }


def _make_study_context(tmp_path: Path) -> StressEthanolCiproGrowthResolvedContext:
    return StressEthanolCiproGrowthResolvedContext(
        study_dir_exists=True,
        requested_study_dir=None,
        resolved_study_dir=tmp_path / "docs" / "studies" / "demo_study",
        study_repo_root=tmp_path,
        study_id="demo_study",
        selection_source="active_registry",
        registry_path=tmp_path / "docs" / "studies" / "index.yaml",
        active_study="demo_study",
        required_paths={},
        missing_required_files=(),
        pipeline_path=tmp_path / "docs" / "studies" / "demo_study" / "pipeline.yaml",
        pipeline_present=True,
        datasets_entries=(),
        study_pipeline={
            "infer": {
                "preferred_model_family": "evo2_20b",
                "supported_model_families": ["evo2_20b", "evo2_7b"],
                "configs": {
                    "anchor_only_20b": tmp_path / "config.anchor_only.evo2_20b.yaml",
                    "anchor_only_7b": tmp_path / "config.anchor_only.evo2_7b.yaml",
                    "full_lane_set_20b": tmp_path / "config.full_lane_set.evo2_20b.yaml",
                },
            }
        },
        canonical_usr_root_path=tmp_path / "usr_root",
        dataset_states=(
            {
                "dataset": "densegen_demo_anchor",
                "declared_status": "present",
                "exists": True,
                "rows": 8,
            },
            {
                "dataset": "promoter/demo_anchor_set",
                "declared_status": "present",
                "exists": True,
                "rows": 8,
            },
            {
                "dataset": "promoter/demo_construct_contexts",
                "declared_status": "present",
                "exists": True,
                "rows": 8,
            },
            {
                "dataset": "usr_demo_opal_candidates",
                "role": "opal_candidate_feature_table",
                "declared_status": "planned",
                "exists": False,
                "rows": None,
            },
        ),
        dataset_index={},
        missing_declared_present=(),
        present_but_planned=(),
        execution_surface_states=(),
        execution_surface_index={
            "infer_batch_20b_with_notify.anchor_only": tmp_path / "runbooks" / "anchor_only_20b.yaml",
            "infer_batch_7b_with_notify.anchor_only": tmp_path / "runbooks" / "anchor_only_7b.yaml",
        },
        missing_execution_surfaces=(),
        phase_states=(
            {"id": "infer_batch_preparation", "status": "in_progress"},
            {
                "id": "infer_anchor_only_20b",
                "status": "planned",
                "next_surface": "runbooks/anchor_only_20b.yaml",
            },
            {
                "id": "infer_anchor_only_7b",
                "status": "planned",
                "next_surface": "runbooks/anchor_only_7b.yaml",
            },
        ),
        current_phase="infer_batch_preparation",
        current_phase_is_known=True,
        next_ready_phase=None,
        next_in_progress_phase={"id": "infer_batch_preparation", "status": "in_progress"},
        next_planned_phase={"id": "infer_anchor_only_20b", "status": "planned"},
        blocked_phases=(),
        densegen_dataset_id="densegen_demo_anchor",
        densegen_rows=8,
        densegen_row_target=10,
        densegen_row_gap=2,
        merged_anchor_dataset_id="promoter/demo_anchor_set",
        merged_anchor_rows=8,
        construct_context_dataset_id="promoter/demo_construct_contexts",
        construct_context_rows=8,
        dataset_refresh_states=(
            {
                "id": "merged_anchor_from_densegen",
                "state": "ok",
                "summary": "Merged anchor dataset is at least as current as the DenseGen source.",
                "upstream_dataset": "densegen_demo_anchor",
                "upstream_rows": 8,
                "downstream_dataset": "promoter/demo_anchor_set",
                "downstream_rows": 8,
                "lag_rows": 0,
            },
            {
                "id": "construct_contexts_from_merged_anchor",
                "state": "ok",
                "summary": "Construct context dataset is at least as current as the merged anchor dataset.",
                "upstream_dataset": "promoter/demo_anchor_set",
                "upstream_rows": 8,
                "downstream_dataset": "promoter/demo_construct_contexts",
                "downstream_rows": 8,
                "lag_rows": 0,
            },
        ),
        stale_dataset_ids=(),
        evidence={"study_id": "demo_study"},
    )


def test_stress_study_downstream_surface_preserves_opal_candidate_feature_table_contract(tmp_path: Path) -> None:
    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        study_id="stress_ethanol_cipro_growth",
        study_pipeline={
            **base_context.study_pipeline,
            "opal": {
                "doc": "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md",
                "config": "src/dnadesign/opal/campaigns/stress_eth_cip/configs/campaign.yaml",
                "state": "configured_pending_candidate_table",
                "candidate_feature_table": {
                    "dataset": "usr_demo_opal_candidates",
                    "role": "opal_candidate_feature_table",
                    "x_column": "latentdna__evo2_7b__context_anchor_mean_bidir_concat",
                    "x_source": "intermediate_embedding_7b_context_anchor_mean_bidir_concat",
                },
            },
        },
        ops_contract=StudyOpsContract(
            study_id="stress_ethanol_cipro_growth",
            status_kind="stress-ethanol-cipro-growth-status",
            preflight_kind="stress-ethanol-cipro-growth-preflight",
            phase_order=("infer_batch_preparation",),
            snapshot_summary_scope="repo",
            preflight=StudyPreflightContract(default_scope="next"),
            current_phase_id="infer_batch_preparation",
            artifacts={
                "opal_candidate_feature_table": {
                    "artifact_type": "dataset",
                    "role": "opal_candidate_feature_table",
                    "dataset_id": "usr_demo_opal_candidates",
                    "ref": "repo:src/dnadesign/usr/datasets/usr_demo_opal_candidates",
                    "x_column": "latentdna__evo2_7b__context_anchor_mean_bidir_concat",
                    "x_source": "intermediate_embedding_7b_context_anchor_mean_bidir_concat",
                }
            },
        ),
    )

    surfaces = inspect_stress_ethanol_cipro_growth_downstream_surfaces(study_context=study_context)

    opal = surfaces["opal"]
    assert opal["configured"] is True
    assert opal["state"] == "configured_pending_candidate_table"
    assert opal["entry_artifact"] == "usr_demo_opal_candidates"
    assert opal["candidate_feature_table"] == {
        "dataset": "usr_demo_opal_candidates",
        "role": "opal_candidate_feature_table",
        "x_column": "latentdna__evo2_7b__context_anchor_mean_bidir_concat",
        "x_source": "intermediate_embedding_7b_context_anchor_mean_bidir_concat",
        "ref": "repo:src/dnadesign/usr/datasets/usr_demo_opal_candidates",
        "resolved_ref": str((tmp_path / "src" / "dnadesign" / "usr" / "datasets" / "usr_demo_opal_candidates")),
    }
    assert "feature_matrix" not in str(opal)


def test_stress_study_downstream_surface_rejects_generic_opal_matrix_role(tmp_path: Path) -> None:
    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        study_id="stress_ethanol_cipro_growth",
        study_pipeline={
            **base_context.study_pipeline,
            "opal": {
                "config": "src/dnadesign/opal/campaigns/stress_eth_cip/configs/campaign.yaml",
                "candidate_feature_table": {
                    "dataset": "usr_demo_opal_candidates",
                    "role": "feature_matrix",
                    "x_column": "latentdna__evo2_7b__context_anchor_mean_bidir_concat",
                },
            },
        },
    )

    with pytest.raises(ValueError, match="candidate_feature_table role must be 'opal_candidate_feature_table'"):
        inspect_stress_ethanol_cipro_growth_downstream_surfaces(study_context=study_context)


def test_resolve_stress_ethanol_cipro_growth_status_context_limits_notify_profiles_to_runtime_lanes(
    tmp_path: Path,
) -> None:
    study_context = _make_study_context(tmp_path)

    def _resolve_named_path_mapping(value, *, repo_root, label, status_kind):
        del repo_root, label, status_kind
        return {name: Path(path) for name, path in dict(value or {}).items()}

    def _resolve_infer_runtime_lane_contracts(config_paths, *, preferred_model_family):
        del preferred_model_family
        return (
            SimpleNamespace(
                phase_id="infer_anchor_only_20b",
                config_label="anchor_only_20b",
                config_path=config_paths["anchor_only_20b"],
                runtime_label="anchor_only_20b",
            ),
            SimpleNamespace(
                phase_id="infer_anchor_only_7b",
                config_label="anchor_only_7b",
                config_path=config_paths["anchor_only_7b"],
                runtime_label="anchor_only_7b",
            ),
        )

    resolved = resolve_stress_ethanol_cipro_growth_status_context(
        study_context=study_context,
        status_kind="stress-ethanol-cipro-growth-status",
        dependencies=StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=StressEthanolCiproGrowthInferRuntimeDependencies(
                resolve_named_path_mapping=_resolve_named_path_mapping,
                resolve_infer_runtime_lane_contracts=_resolve_infer_runtime_lane_contracts,
                derive_infer_notify_profile_paths=lambda config_paths: (
                    {label: path.parent / "notify" / f"{label}.json" for label, path in config_paths.items()},
                    {},
                ),
                load_infer_model_summary=lambda config_path: {
                    "model_id": f"model-{config_path.stem}",
                    "device": "cuda:0",
                },
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda *, phase_id, model_family: bool(
                model_family and model_family in phase_id
            ),
            inspect_semantic_completeness=lambda **kwargs: None,
        ),
    )

    assert tuple(resolved.infer_runtime.infer_notify_profile_paths) == ("anchor_only_20b", "anchor_only_7b")
    assert "full_lane_set_20b" not in resolved.infer_runtime.infer_notify_profile_paths
    assert [summary.label for summary in resolved.infer_runtime.runtime_model_summaries] == [
        "anchor_only_20b",
        "anchor_only_7b",
    ]


def test_build_stress_ethanol_cipro_growth_status_preserves_summary_and_attention_contract(tmp_path: Path) -> None:
    study_context = _make_study_context(tmp_path)
    status_context = StressEthanolCiproGrowthStatusResolvedContext(
        infer_runtime=StressEthanolCiproGrowthInferRuntimeResolvedContext(
            preferred_model_family="evo2_20b",
            supported_model_families=("evo2_20b", "evo2_7b"),
            infer_config_paths={},
            runtime_lane_contracts=(),
            runtime_config_paths={},
            phase_targets=(),
            phase_targets_by_id={},
            config_phase_ids={},
            runtime_phase_ids={},
            infer_notify_profile_paths={},
            infer_notify_profile_errors={},
            runtime_model_summaries=(
                SimpleNamespace(
                    label="anchor_only_20b",
                    model_id="evo2_20b",
                    device="cuda:0",
                    as_dict=lambda: {
                        "label": "anchor_only_20b",
                        "model_id": "evo2_20b",
                        "device": "cuda:0",
                    },
                ),
            ),
            gpu_required_runtime_labels=("anchor_only_20b",),
        ),
    )

    state, summary, evidence = build_stress_ethanol_cipro_growth_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=StressEthanolCiproGrowthInferRuntimeDependencies(
                resolve_named_path_mapping=lambda *args, **kwargs: {},
                resolve_infer_runtime_lane_contracts=lambda *args, **kwargs: (),
                derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
                load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cuda:0"},
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda *, phase_id, model_family: bool(
                model_family and model_family in phase_id
            ),
            inspect_semantic_completeness=lambda **kwargs: None,
        ),
        summary_scope="repo",
    )

    assert state == "attention"
    assert summary == (
        "demo_study: phase infer_batch_preparation; preferred infer evo2_20b; "
        "source gate active densegen_demo_anchor 8/10 rows (gap=2); "
        "handoffs ready anchor=8 construct=8; "
        "next in_progress infer_batch_preparation"
    )
    assert evidence["attention_reasons"] == ["DenseGen source gate is still active"]
    assert evidence["source_growth_state"] == {
        "state": "attention",
        "dataset": "densegen_demo_anchor",
        "current_rows": 8,
        "target_rows": 10,
        "gap_rows": 2,
        "target_met": False,
        "gates_current_phase": True,
        "source_phase_id": None,
        "source_phase_status": None,
        "superseded_by_handoffs": False,
        "max_handoff_rows": 8,
        "drives_top_level_attention": True,
        "summary": "source gate active densegen_demo_anchor 8/10 rows (gap=2)",
    }
    assert evidence["handoff_readiness_state"] == {
        "state": "ok",
        "pending_datasets": [],
        "stale_datasets": [],
        "drives_top_level_attention": False,
        "summary": "handoffs ready anchor=8 construct=8",
    }
    assert evidence["planned_outputs_state"] == {
        "state": "ok",
        "pending_datasets": ["usr_demo_opal_candidates"],
        "drives_top_level_attention": False,
        "include_in_summary": False,
        "summary": "future outputs still planned usr_demo_opal_candidates",
    }
    assert evidence["semantic_completeness_state"] is None
    assert evidence["latentdna"]["state"] == "not_configured"
    assert evidence["cluster"]["state"] == "planned"
    assert evidence["opal"]["state"] == "not_configured"
    assert evidence["analysis_surfaces"] == {}
    assert "local_advisories" not in evidence


def test_build_status_fails_closed_on_opal_round_receipt_drift(tmp_path: Path) -> None:
    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        densegen_row_target=8,
        densegen_row_gap=0,
    )
    status_context = StressEthanolCiproGrowthStatusResolvedContext(
        infer_runtime=StressEthanolCiproGrowthInferRuntimeResolvedContext(
            preferred_model_family=None,
            supported_model_families=(),
            infer_config_paths={},
            runtime_lane_contracts=(),
            runtime_config_paths={},
            phase_targets=(),
            phase_targets_by_id={},
            config_phase_ids={},
            runtime_phase_ids={},
            infer_notify_profile_paths={},
            infer_notify_profile_errors={},
            runtime_model_summaries=(),
            gpu_required_runtime_labels=(),
        )
    )
    receipt_summary = "OPAL round-0 run receipt has 1 integrity mismatch(es)"

    state, summary, evidence = build_stress_ethanol_cipro_growth_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=StressEthanolCiproGrowthInferRuntimeDependencies(
                resolve_named_path_mapping=lambda *args, **kwargs: {},
                resolve_infer_runtime_lane_contracts=lambda *args, **kwargs: (),
                derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
                load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cpu"},
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda **kwargs: False,
            inspect_semantic_completeness=lambda **kwargs: None,
            inspect_additional_downstream_surfaces=lambda **kwargs: {
                "cluster": {"configured": False, "state": "planned"},
                "opal": {
                    "configured": True,
                    "state": "round0_candidate_review",
                    "integrity_state": "attention",
                    "run_receipt": {
                        "configured": True,
                        "state": "attention",
                        "drives_top_level_attention": True,
                        "summary": receipt_summary,
                        "mismatches": [
                            {
                                "field": "artifacts.selection_batch.actual_sha256",
                                "expected": "expected",
                                "actual": "actual",
                            }
                        ],
                    },
                },
            },
        ),
        summary_scope="repo",
    )

    assert state == "attention"
    assert receipt_summary in summary
    assert "OPAL round-0 run receipt integrity is not ok" in evidence["attention_reasons"]


def test_build_stress_ethanol_cipro_growth_status_demotes_source_gate_once_handoffs_exceed_target(
    tmp_path: Path,
) -> None:
    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        dataset_states=(
            {
                "dataset": "densegen_demo_anchor",
                "declared_status": "present",
                "exists": True,
                "rows": 8,
            },
            {
                "dataset": "promoter/demo_anchor_set",
                "declared_status": "present",
                "exists": True,
                "rows": 12,
            },
            {
                "dataset": "promoter/demo_construct_contexts",
                "declared_status": "present",
                "exists": True,
                "rows": 12,
            },
            {
                "dataset": "usr_demo_opal_candidates",
                "role": "opal_candidate_feature_table",
                "declared_status": "planned",
                "exists": False,
                "rows": None,
            },
        ),
        phase_states=(
            {
                "id": "densegen_growth",
                "status": "parallel_optional",
                "primary_dataset": "densegen_demo_anchor",
            },
            {"id": "infer_batch_preparation", "status": "in_progress"},
            {
                "id": "infer_anchor_only_20b",
                "status": "planned",
                "next_surface": "runbooks/anchor_only_20b.yaml",
            },
        ),
        merged_anchor_rows=12,
        construct_context_rows=12,
        dataset_refresh_states=(
            {
                "id": "merged_anchor_from_densegen",
                "state": "ok",
                "summary": "Merged anchor dataset is at least as current as the DenseGen source.",
                "upstream_dataset": "densegen_demo_anchor",
                "upstream_rows": 8,
                "downstream_dataset": "promoter/demo_anchor_set",
                "downstream_rows": 12,
                "lag_rows": 0,
            },
            {
                "id": "construct_contexts_from_merged_anchor",
                "state": "ok",
                "summary": "Construct context dataset is at least as current as the merged anchor dataset.",
                "upstream_dataset": "promoter/demo_anchor_set",
                "upstream_rows": 12,
                "downstream_dataset": "promoter/demo_construct_contexts",
                "downstream_rows": 12,
                "lag_rows": 0,
            },
        ),
    )
    status_context = StressEthanolCiproGrowthStatusResolvedContext(
        infer_runtime=StressEthanolCiproGrowthInferRuntimeResolvedContext(
            preferred_model_family="evo2_20b",
            supported_model_families=("evo2_20b", "evo2_7b"),
            infer_config_paths={},
            runtime_lane_contracts=(),
            runtime_config_paths={},
            phase_targets=(),
            phase_targets_by_id={},
            config_phase_ids={},
            runtime_phase_ids={},
            infer_notify_profile_paths={},
            infer_notify_profile_errors={},
            runtime_model_summaries=(),
            gpu_required_runtime_labels=(),
        ),
    )

    state, summary, evidence = build_stress_ethanol_cipro_growth_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=StressEthanolCiproGrowthInferRuntimeDependencies(
                resolve_named_path_mapping=lambda *args, **kwargs: {},
                resolve_infer_runtime_lane_contracts=lambda *args, **kwargs: (),
                derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
                load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cuda:0"},
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda *, phase_id, model_family: bool(
                model_family and model_family in phase_id
            ),
            inspect_semantic_completeness=lambda **kwargs: None,
        ),
        summary_scope="repo",
    )

    assert state == "ok"
    assert summary == (
        "demo_study: phase infer_batch_preparation; preferred infer evo2_20b; "
        "handoffs ready anchor=12 construct=12; "
        "source gate superseded by downstream handoffs densegen_demo_anchor 8/10 rows (gap=2); "
        "next in_progress infer_batch_preparation"
    )
    assert "attention_reasons" not in evidence
    assert evidence["source_growth_state"]["state"] == "ok"
    assert evidence["source_growth_state"]["target_met"] is False
    assert evidence["source_growth_state"]["gates_current_phase"] is False
    assert evidence["source_growth_state"]["source_phase_id"] == "densegen_growth"
    assert evidence["source_growth_state"]["source_phase_status"] == "parallel_optional"
    assert evidence["source_growth_state"]["superseded_by_handoffs"] is True
    assert evidence["source_growth_state"]["max_handoff_rows"] == 12


def test_build_stress_ethanol_cipro_growth_status_surfaces_semantic_completeness_attention(tmp_path: Path) -> None:
    study_context = _make_study_context(tmp_path)
    status_context = StressEthanolCiproGrowthStatusResolvedContext(
        infer_runtime=StressEthanolCiproGrowthInferRuntimeResolvedContext(
            preferred_model_family="evo2_20b",
            supported_model_families=("evo2_20b", "evo2_7b"),
            infer_config_paths={},
            runtime_lane_contracts=(),
            runtime_config_paths={},
            phase_targets=(),
            phase_targets_by_id={},
            config_phase_ids={},
            runtime_phase_ids={},
            infer_notify_profile_paths={},
            infer_notify_profile_errors={},
            runtime_model_summaries=(),
            gpu_required_runtime_labels=(),
        ),
    )

    semantic_state = {
        "state": "attention",
        "drives_top_level_attention": True,
        "summary": (
            "source overlay needs compaction densegen_demo_anchor:densegen; "
            "anchor DenseGen metadata incomplete promoter/demo_anchor_set 6/8"
        ),
    }

    state, summary, evidence = build_stress_ethanol_cipro_growth_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=StressEthanolCiproGrowthInferRuntimeDependencies(
                resolve_named_path_mapping=lambda *args, **kwargs: {},
                resolve_infer_runtime_lane_contracts=lambda *args, **kwargs: (),
                derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
                load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cuda:0"},
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda *, phase_id, model_family: bool(
                model_family and model_family in phase_id
            ),
            inspect_semantic_completeness=lambda **kwargs: semantic_state,
        ),
        summary_scope="repo",
    )

    assert state == "attention"
    assert "source overlay needs compaction" in summary
    assert evidence["semantic_completeness_state"] == semantic_state
    assert "shared handoff metadata is semantically incomplete" in evidence["attention_reasons"]


def test_build_stress_ethanol_cipro_growth_status_surfaces_configured_latentdna_readiness_attention(
    tmp_path: Path,
) -> None:
    study_context = replace(
        _make_study_context(tmp_path),
        densegen_row_target=8,
        densegen_row_gap=0,
    )
    status_context = StressEthanolCiproGrowthStatusResolvedContext(
        infer_runtime=StressEthanolCiproGrowthInferRuntimeResolvedContext(
            preferred_model_family="evo2_7b",
            supported_model_families=("evo2_7b",),
            infer_config_paths={},
            runtime_lane_contracts=(),
            runtime_config_paths={},
            phase_targets=(),
            phase_targets_by_id={},
            config_phase_ids={},
            runtime_phase_ids={},
            infer_notify_profile_paths={},
            infer_notify_profile_errors={},
            runtime_model_summaries=(),
            gpu_required_runtime_labels=(),
        ),
    )
    latentdna_state = {
        "configured": True,
        "state": "error",
        "summary": "LatentDNA snapshot unreadable",
    }

    state, summary, evidence = build_stress_ethanol_cipro_growth_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=StressEthanolCiproGrowthInferRuntimeDependencies(
                resolve_named_path_mapping=lambda *args, **kwargs: {},
                resolve_infer_runtime_lane_contracts=lambda *args, **kwargs: (),
                derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
                load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cuda:0"},
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda *, phase_id, model_family: bool(
                model_family and model_family in phase_id
            ),
            inspect_semantic_completeness=lambda **kwargs: None,
            inspect_latentdna_readiness=lambda **kwargs: latentdna_state,
        ),
        summary_scope="repo",
    )

    assert state == "attention"
    assert "LatentDNA snapshot unreadable" in summary
    assert evidence["latentdna"] == latentdna_state
    assert "LatentDNA readiness is not ok" in evidence["attention_reasons"]


def test_build_stress_ethanol_cipro_growth_status_surfaces_missing_latentdna_sources_in_summary(
    tmp_path: Path,
) -> None:
    study_context = replace(
        _make_study_context(tmp_path),
        densegen_row_target=8,
        densegen_row_gap=0,
    )
    status_context = StressEthanolCiproGrowthStatusResolvedContext(
        infer_runtime=StressEthanolCiproGrowthInferRuntimeResolvedContext(
            preferred_model_family="evo2_7b",
            supported_model_families=("evo2_7b",),
            infer_config_paths={},
            runtime_lane_contracts=(),
            runtime_config_paths={},
            phase_targets=(),
            phase_targets_by_id={},
            config_phase_ids={},
            runtime_phase_ids={},
            infer_notify_profile_paths={},
            infer_notify_profile_errors={},
            runtime_model_summaries=(),
            gpu_required_runtime_labels=(),
        ),
    )
    latentdna_state = {
        "configured": True,
        "state": "attention",
        "summary": "LatentDNA primary readiness attention: missing source aliases: reference_core60",
        "missing_source_datasets": ["reference_core60"],
        "missing_appendix_source_datasets": ["regulondb_native_core60"],
        "appendix_state": "attention",
    }

    state, summary, evidence = build_stress_ethanol_cipro_growth_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=StressEthanolCiproGrowthInferRuntimeDependencies(
                resolve_named_path_mapping=lambda *args, **kwargs: {},
                resolve_infer_runtime_lane_contracts=lambda *args, **kwargs: (),
                derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
                load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cuda:0"},
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda *, phase_id, model_family: bool(
                model_family and model_family in phase_id
            ),
            inspect_semantic_completeness=lambda **kwargs: None,
            inspect_latentdna_readiness=lambda **kwargs: latentdna_state,
        ),
        summary_scope="repo",
    )

    assert state == "attention"
    assert "missing source aliases: reference_core60" in summary
    assert evidence["latentdna"]["missing_source_datasets"] == ["reference_core60"]
    assert evidence["latentdna"]["missing_appendix_source_datasets"] == ["regulondb_native_core60"]


def test_build_stress_ethanol_cipro_growth_status_keeps_latentdna_appendix_drift_nonblocking(
    tmp_path: Path,
) -> None:
    study_context = replace(
        _make_study_context(tmp_path),
        densegen_row_target=8,
        densegen_row_gap=0,
    )
    status_context = StressEthanolCiproGrowthStatusResolvedContext(
        infer_runtime=StressEthanolCiproGrowthInferRuntimeResolvedContext(
            preferred_model_family="evo2_7b",
            supported_model_families=("evo2_7b",),
            infer_config_paths={},
            runtime_lane_contracts=(),
            runtime_config_paths={},
            phase_targets=(),
            phase_targets_by_id={},
            config_phase_ids={},
            runtime_phase_ids={},
            infer_notify_profile_paths={},
            infer_notify_profile_errors={},
            runtime_model_summaries=(),
            gpu_required_runtime_labels=(),
        ),
    )
    latentdna_state = {
        "configured": True,
        "state": "ok",
        "summary": "LatentDNA primary readiness ok.",
        "missing_source_datasets": [],
        "missing_appendix_source_datasets": ["regulondb_native_core60"],
        "appendix_state": "attention",
    }

    state, summary, evidence = build_stress_ethanol_cipro_growth_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=StressEthanolCiproGrowthInferRuntimeDependencies(
                resolve_named_path_mapping=lambda *args, **kwargs: {},
                resolve_infer_runtime_lane_contracts=lambda *args, **kwargs: (),
                derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
                load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cuda:0"},
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda *, phase_id, model_family: bool(
                model_family and model_family in phase_id
            ),
            inspect_semantic_completeness=lambda **kwargs: None,
            inspect_latentdna_readiness=lambda **kwargs: latentdna_state,
        ),
        summary_scope="repo",
    )

    assert state == "ok"
    assert "LatentDNA primary readiness ok." not in summary
    assert "attention_reasons" not in evidence
    assert evidence["latentdna"]["missing_appendix_source_datasets"] == ["regulondb_native_core60"]


def test_build_stress_ethanol_cipro_growth_status_surfaces_sequence_view_and_feature_completion_sections(
    tmp_path: Path,
) -> None:
    study_context = replace(
        _make_study_context(tmp_path),
        densegen_row_target=8,
        densegen_row_gap=0,
    )
    status_context = StressEthanolCiproGrowthStatusResolvedContext(
        infer_runtime=StressEthanolCiproGrowthInferRuntimeResolvedContext(
            preferred_model_family="evo2_7b",
            supported_model_families=("evo2_7b",),
            infer_config_paths={},
            runtime_lane_contracts=(),
            runtime_config_paths={},
            phase_targets=(),
            phase_targets_by_id={},
            config_phase_ids={},
            runtime_phase_ids={},
            infer_notify_profile_paths={},
            infer_notify_profile_errors={},
            runtime_model_summaries=(),
            gpu_required_runtime_labels=(),
        ),
    )
    sequence_view_contract_state = {
        "state": "attention",
        "drives_top_level_attention": True,
        "summary": "sequence-view product contracts 1/2 ok; required_failures=1 optional_failures=0",
        "checks": [
            {
                "check_id": "infer.sequence_views.context_contract",
                "dataset": "promoter/demo_construct_contexts",
                "state": "attention",
                "required": True,
            }
        ],
    }
    infer_feature_completion_state = {
        "state": "attention",
        "drives_top_level_attention": False,
        "summary": (
            "infer sequence-view feature completion reusable_vectors=1 stale_vectors=0 "
            "missing_vectors=2 reusable_scalars=0 stale_scalars=0 missing_scalars=2 missing_products=0"
        ),
        "aggregate": {
            "reusable_vectors": 1,
            "reusable_scalars": 0,
            "stale_vectors": 0,
            "stale_scalars": 0,
            "missing_vectors": 2,
            "missing_scalars": 2,
            "missing_products": 0,
            "counts_by_product_kind": {"construct_insert": 2},
            "counts_by_orientation": {"forward": 2},
            "counts_by_pooling_operation": {"seq_mean": 2},
        },
    }

    state, summary, evidence = build_stress_ethanol_cipro_growth_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=StressEthanolCiproGrowthInferRuntimeDependencies(
                resolve_named_path_mapping=lambda *args, **kwargs: {},
                resolve_infer_runtime_lane_contracts=lambda *args, **kwargs: (),
                derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
                load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cuda:0"},
                string_or_none=_string_or_none,
                string_list_or_empty=_string_list_or_empty,
            ),
            phase_matches_infer_model_family=lambda *, phase_id, model_family: bool(
                model_family and model_family in phase_id
            ),
            inspect_semantic_completeness=lambda **kwargs: None,
            inspect_sequence_view_contracts=lambda **kwargs: sequence_view_contract_state,
            inspect_infer_feature_completion=lambda **kwargs: infer_feature_completion_state,
        ),
        summary_scope="repo",
    )

    assert state == "attention"
    assert "sequence-view product contracts 1/2 ok" in summary
    assert "infer sequence-view feature completion reusable_vectors=1 stale_vectors=0" in summary
    assert "missing_vectors=2 reusable_scalars=0 stale_scalars=0 missing_scalars=2 missing_products=0" in summary
    assert evidence["sequence_view_contract_state"] == sequence_view_contract_state
    assert evidence["infer_feature_completion_state"] == infer_feature_completion_state
    assert "sequence-view product contracts are incomplete" in evidence["attention_reasons"]
    assert "Infer feature completion is incomplete" not in evidence["attention_reasons"]


def test_stress_ethanol_cipro_snapshot_service_keeps_deep_host_and_feature_probes_out_of_record_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    study_context = _make_study_context(tmp_path)
    contract = StudyOpsContract(
        study_id="demo_study",
        status_kind="stress-ethanol-cipro-growth-status",
        preflight_kind="stress-ethanol-cipro-growth-preflight",
        phase_order=("infer_batch_preparation", "infer_anchor_only_20b", "infer_anchor_only_7b"),
        snapshot_summary_scope="repo",
        preflight=StudyPreflightContract(default_scope="next"),
        current_phase_id="infer_batch_preparation",
        phases=(),
        raw_payload={
            "study_id": "demo_study",
            "ops_surfaces": {
                "status_kind": "stress-ethanol-cipro-growth-status",
                "preflight_kind": "stress-ethanol-cipro-growth-preflight",
            },
        },
    )
    service_context = StudyStatusContext(
        repo_root=tmp_path,
        study_root=study_context.resolved_study_dir,
        contract=contract,
        service_context=StressEthanolCiproGrowthStatusServiceContext(study_context=study_context),
    )

    def _forbidden_probe() -> dict[str, object]:
        raise AssertionError("cheap snapshot must not probe local GPU inventory")

    monkeypatch.setattr(
        "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
        _forbidden_probe,
    )
    monkeypatch.setattr(
        "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.build_stress_ethanol_cipro_growth_infer_runtime_dependencies",
        lambda: StressEthanolCiproGrowthInferRuntimeDependencies(
            resolve_named_path_mapping=lambda value, *, repo_root, label, status_kind: {
                name: Path(path) for name, path in dict(value or {}).items()
            },
            resolve_infer_runtime_lane_contracts=lambda config_paths, *, preferred_model_family: (
                SimpleNamespace(
                    phase_id="infer_anchor_only_20b",
                    config_label="anchor_only_20b",
                    config_path=config_paths["anchor_only_20b"],
                    runtime_label="anchor_only_20b",
                ),
                SimpleNamespace(
                    phase_id="infer_anchor_only_7b",
                    config_label="anchor_only_7b",
                    config_path=config_paths["anchor_only_7b"],
                    runtime_label="anchor_only_7b",
                ),
            ),
            derive_infer_notify_profile_paths=lambda config_paths: (
                {label: path.parent / "notify" / f"{label}.json" for label, path in config_paths.items()},
                {},
            ),
            load_infer_model_summary=lambda config_path: {
                "model_id": f"model-{Path(config_path).stem}",
                "device": "cuda:0",
            },
            string_or_none=_string_or_none,
            string_list_or_empty=_string_list_or_empty,
        ),
    )

    state, summary, evidence = STUDY_STATUS_SERVICE.build_snapshot(service_context)

    assert state == "attention"
    assert "preferred infer evo2_20b" in summary
    assert "infer_local_gpu_inventory" not in evidence
    assert evidence["infer_feature_completion_state"] is None
