"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_promoter_snapshot.py

Focused tests for the study-owned snapshot coordination layer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from dnadesign.studies.core.models import StudyOpsContract, StudyPreflightContract, StudyStatusContext
from dnadesign.studies.families.promoter.adapter import STUDY_FAMILY_ADAPTER, PromoterFamilyContext
from dnadesign.studies.families.promoter.infer_runtime import (
    PromoterStudyInferRuntimeDependencies,
    PromoterStudyInferRuntimeResolvedContext,
)
from dnadesign.studies.families.promoter.record_normalizer import PromoterStudyResolvedContext
from dnadesign.studies.families.promoter.snapshot import (
    PromoterStudyStatusDependencies,
    PromoterStudyStatusResolvedContext,
    build_promoter_study_status,
    resolve_promoter_study_status_context,
)


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


def _make_study_context(tmp_path: Path) -> PromoterStudyResolvedContext:
    return PromoterStudyResolvedContext(
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
                "dataset": "densegen/demo_anchor",
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
                "dataset": "promoter/demo_feature_matrix",
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
        densegen_dataset_id="densegen/demo_anchor",
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
                "upstream_dataset": "densegen/demo_anchor",
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


def test_resolve_promoter_study_status_context_limits_notify_profiles_to_runtime_lanes(tmp_path: Path) -> None:
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

    resolved = resolve_promoter_study_status_context(
        study_context=study_context,
        status_kind="promoter-study-status",
        dependencies=PromoterStudyStatusDependencies(
            infer_runtime=PromoterStudyInferRuntimeDependencies(
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


def test_build_promoter_study_status_preserves_summary_and_attention_contract(tmp_path: Path) -> None:
    study_context = _make_study_context(tmp_path)
    status_context = PromoterStudyStatusResolvedContext(
        infer_runtime=PromoterStudyInferRuntimeResolvedContext(
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

    state, summary, evidence = build_promoter_study_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=PromoterStudyStatusDependencies(
            infer_runtime=PromoterStudyInferRuntimeDependencies(
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
        "source gate active densegen/demo_anchor 8/10 rows (gap=2); "
        "handoffs ready anchor=8 construct=8; "
        "next in_progress infer_batch_preparation"
    )
    assert evidence["attention_reasons"] == ["DenseGen source gate is still active"]
    assert evidence["source_growth_state"] == {
        "state": "attention",
        "dataset": "densegen/demo_anchor",
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
        "summary": "source gate active densegen/demo_anchor 8/10 rows (gap=2)",
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
        "pending_datasets": ["promoter/demo_feature_matrix"],
        "drives_top_level_attention": False,
        "include_in_summary": False,
        "summary": "future outputs still planned promoter/demo_feature_matrix",
    }
    assert evidence["semantic_completeness_state"] is None
    assert evidence["latentdna"]["state"] == "not_configured"
    assert evidence["cluster"]["state"] == "planned"
    assert evidence["opal"]["state"] == "not_configured"
    assert "local_advisories" not in evidence


def test_build_promoter_study_status_demotes_source_gate_once_handoffs_exceed_target(tmp_path: Path) -> None:
    base_context = _make_study_context(tmp_path)
    study_context = replace(
        base_context,
        dataset_states=(
            {
                "dataset": "densegen/demo_anchor",
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
                "dataset": "promoter/demo_feature_matrix",
                "declared_status": "planned",
                "exists": False,
                "rows": None,
            },
        ),
        phase_states=(
            {
                "id": "densegen_growth",
                "status": "parallel_optional",
                "primary_dataset": "densegen/demo_anchor",
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
                "upstream_dataset": "densegen/demo_anchor",
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
    status_context = PromoterStudyStatusResolvedContext(
        infer_runtime=PromoterStudyInferRuntimeResolvedContext(
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

    state, summary, evidence = build_promoter_study_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=PromoterStudyStatusDependencies(
            infer_runtime=PromoterStudyInferRuntimeDependencies(
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
        "source gate superseded by downstream handoffs densegen/demo_anchor 8/10 rows (gap=2); "
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


def test_build_promoter_study_status_surfaces_semantic_completeness_attention(tmp_path: Path) -> None:
    study_context = _make_study_context(tmp_path)
    status_context = PromoterStudyStatusResolvedContext(
        infer_runtime=PromoterStudyInferRuntimeResolvedContext(
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
            "source overlay needs compaction densegen/demo_anchor:densegen; "
            "anchor DenseGen metadata incomplete promoter/demo_anchor_set 6/8"
        ),
    }

    state, summary, evidence = build_promoter_study_status(
        study_context=study_context,
        status_context=status_context,
        dependencies=PromoterStudyStatusDependencies(
            infer_runtime=PromoterStudyInferRuntimeDependencies(
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


def test_promoter_snapshot_adapter_never_touches_local_gpu_probe(tmp_path: Path, monkeypatch) -> None:
    study_context = _make_study_context(tmp_path)
    contract = StudyOpsContract(
        study_id="demo_study",
        family="promoter",
        phase_order=("infer_batch_preparation", "infer_anchor_only_20b", "infer_anchor_only_7b"),
        snapshot_summary_scope="repo",
        preflight=StudyPreflightContract(default_scope="next"),
        current_phase_id="infer_batch_preparation",
        phases=(),
        raw_payload={"study_id": "demo_study", "family": "promoter"},
    )
    adapter_context = StudyStatusContext(
        repo_root=tmp_path,
        study_root=study_context.resolved_study_dir,
        contract=contract,
        family_context=PromoterFamilyContext(study_context=study_context),
    )

    def _forbidden_probe() -> dict[str, object]:
        raise AssertionError("cheap snapshot must not probe local GPU inventory")

    monkeypatch.setattr(
        "dnadesign.studies.families.promoter.adapter.inspect_local_infer_gpu_inventory",
        _forbidden_probe,
    )
    monkeypatch.setattr(
        "dnadesign.studies.families.promoter.adapter.build_promoter_study_infer_runtime_dependencies",
        lambda: PromoterStudyInferRuntimeDependencies(
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

    state, summary, evidence = STUDY_FAMILY_ADAPTER.build_snapshot(adapter_context)

    assert state == "attention"
    assert "preferred infer evo2_20b" in summary
    assert "infer_local_gpu_inventory" not in evidence
