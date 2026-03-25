"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_promoter_snapshot.py

Focused tests for the study-owned snapshot coordination layer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dnadesign.studies.core.models import StudyOpsContract, StudyPreflightContract, StudyStatusContext
from dnadesign.studies.promoter.context import PromoterStudyResolvedContext
from dnadesign.studies.promoter.family import PROMOTER_STUDY_ADAPTER, PromoterFamilyContext
from dnadesign.studies.promoter.infer_runtime import (
    PromoterStudyInferRuntimeDependencies,
    PromoterStudyInferRuntimeResolvedContext,
)
from dnadesign.studies.promoter.snapshot import (
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
        resolved_study_dir=tmp_path / "docs" / "studies" / "promoter" / "demo_study",
        study_repo_root=tmp_path,
        study_id="demo_study",
        selection_source="active_registry",
        registry_path=tmp_path / "docs" / "studies" / "promoter" / "index.yaml",
        active_study="demo_study",
        required_paths={},
        missing_required_files=(),
        pipeline_path=tmp_path / "docs" / "studies" / "promoter" / "demo_study" / "pipeline.yaml",
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
        ),
        summary_scope="repo",
    )

    assert state == "attention"
    assert summary == (
        "demo_study: phase infer_batch_preparation; preferred infer evo2_20b; "
        "densegen/demo_anchor 8/10 rows; pending promoter/demo_feature_matrix; "
        "next in_progress infer_batch_preparation"
    )
    assert evidence["attention_reasons"] == [
        "DenseGen anchor target not met",
        "study is not complete",
    ]
    assert "local_advisories" not in evidence


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
        "dnadesign.studies.promoter.family.inspect_local_infer_gpu_inventory",
        _forbidden_probe,
    )
    monkeypatch.setattr(
        "dnadesign.studies.promoter.family.build_promoter_study_infer_runtime_dependencies",
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

    state, summary, evidence = PROMOTER_STUDY_ADAPTER.build_snapshot(adapter_context)

    assert state == "attention"
    assert "preferred infer evo2_20b" in summary
    assert "infer_local_gpu_inventory" not in evidence
