"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/status/test_infer_runtime.py

Focused tests for study-owned infer-runtime projection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dnadesign.studies.units.stress_ethanol_cipro_growth.status.infer_runtime import (
    StressEthanolCiproGrowthInferRuntimeDependencies,
    resolve_stress_ethanol_cipro_growth_infer_runtime_context,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.status.record_normalizer import (
    StressEthanolCiproGrowthResolvedContext,
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
        dataset_states=(),
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
        next_in_progress_phase=None,
        next_planned_phase=None,
        blocked_phases=(),
        densegen_dataset_id=None,
        densegen_rows=None,
        densegen_row_target=None,
        densegen_row_gap=None,
        merged_anchor_dataset_id=None,
        merged_anchor_rows=None,
        construct_context_dataset_id=None,
        construct_context_rows=None,
        dataset_refresh_states=(),
        stale_dataset_ids=(),
        evidence={"study_id": "demo_study"},
    )


def test_resolve_stress_ethanol_cipro_growth_infer_runtime_context_projects_runtime_lanes_once(tmp_path: Path) -> None:
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

    resolved = resolve_stress_ethanol_cipro_growth_infer_runtime_context(
        study_context=study_context,
        status_kind="stress-ethanol-cipro-growth-status",
        dependencies=StressEthanolCiproGrowthInferRuntimeDependencies(
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
    )

    assert tuple(resolved.infer_config_paths) == (
        "anchor_only_20b",
        "anchor_only_7b",
        "full_lane_set_20b",
    )
    assert tuple(resolved.runtime_config_paths) == ("anchor_only_20b", "anchor_only_7b")
    assert tuple(resolved.config_phase_ids.items()) == (
        ("anchor_only_20b", "infer_anchor_only_20b"),
        ("anchor_only_7b", "infer_anchor_only_7b"),
    )
    assert tuple(resolved.runtime_phase_ids.items()) == (
        ("anchor_only_20b", "infer_anchor_only_20b"),
        ("anchor_only_7b", "infer_anchor_only_7b"),
    )
    assert [target.runbook_surface_label for target in resolved.phase_targets] == [
        "infer_batch_20b_with_notify.anchor_only",
        "infer_batch_7b_with_notify.anchor_only",
    ]
    assert tuple(resolved.infer_notify_profile_paths) == ("anchor_only_20b", "anchor_only_7b")
    assert [summary.label for summary in resolved.runtime_model_summaries] == [
        "anchor_only_20b",
        "anchor_only_7b",
    ]
    assert resolved.gpu_required_runtime_labels == ("anchor_only_20b", "anchor_only_7b")


def test_resolve_stress_ethanol_cipro_growth_infer_runtime_context_fails_fast_on_missing_study_surface_mapping(
    tmp_path: Path,
) -> None:
    study_context = _make_study_context(tmp_path)
    study_context = StressEthanolCiproGrowthResolvedContext(
        **{
            **study_context.__dict__,
            "execution_surface_index": {
                "infer_batch_7b_with_notify.anchor_only": tmp_path / "runbooks" / "anchor_only_7b.yaml",
            },
        }
    )

    def _resolve_named_path_mapping(value, *, repo_root, label, status_kind):
        del repo_root, label, status_kind
        return {name: Path(path) for name, path in dict(value or {}).items()}

    resolved_dependencies = StressEthanolCiproGrowthInferRuntimeDependencies(
        resolve_named_path_mapping=_resolve_named_path_mapping,
        resolve_infer_runtime_lane_contracts=lambda config_paths, *, preferred_model_family: (
            SimpleNamespace(
                phase_id="infer_anchor_only_20b",
                config_label="anchor_only_20b",
                config_path=config_paths["anchor_only_20b"],
                runtime_label="anchor_only_20b",
            ),
        ),
        derive_infer_notify_profile_paths=lambda config_paths: ({}, {}),
        load_infer_model_summary=lambda config_path: {"model_id": "demo", "device": "cuda:0"},
        string_or_none=_string_or_none,
        string_list_or_empty=_string_list_or_empty,
    )

    try:
        resolve_stress_ethanol_cipro_growth_infer_runtime_context(
            study_context=study_context,
            status_kind="stress-ethanol-cipro-growth-status",
            dependencies=resolved_dependencies,
        )
    except ValueError as exc:
        assert "not declared under study execution_surfaces" in str(exc)
    else:
        raise AssertionError("expected study-owned infer phase target resolution to fail visibly")
