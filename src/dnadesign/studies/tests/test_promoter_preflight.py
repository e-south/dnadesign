"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_promoter_preflight.py

Focused tests for the study-owned preflight context coordination layer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dnadesign.studies.core.models import (
    StudyOpsContract,
    StudyPhaseContract,
    StudyPreflightContract,
    StudyPreflightNextScopeContract,
)
from dnadesign.studies.promoter.context import PromoterStudyResolvedContext
from dnadesign.studies.promoter.infer_runtime import PromoterStudyInferRuntimeDependencies
from dnadesign.studies.promoter.preflight import (
    PromoterPreflightContextDependencies,
    resolve_promoter_preflight_context,
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


def test_resolve_promoter_preflight_context_projects_infer_batch_targets_in_phase_order(tmp_path: Path) -> None:
    study_repo_root = tmp_path
    infer_configs = {
        "anchor_only_20b": tmp_path / "config.anchor_only.evo2_20b.yaml",
        "anchor_plus_template_20b": tmp_path / "config.anchor_plus_template.evo2_20b.yaml",
        "anchor_only_7b": tmp_path / "config.anchor_only.evo2_7b.yaml",
    }
    study_context = PromoterStudyResolvedContext(
        study_dir_exists=True,
        requested_study_dir=None,
        resolved_study_dir=tmp_path / "docs" / "studies" / "promoter" / "demo_study",
        study_repo_root=study_repo_root,
        study_id="demo_study",
        selection_source="explicit",
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
                "configs": infer_configs,
            }
        },
        canonical_usr_root_path=None,
        dataset_states=(),
        dataset_index={},
        missing_declared_present=(),
        present_but_planned=(),
        execution_surface_states=(),
        execution_surface_index={
            "infer_batch_7b_with_notify.anchor_only": tmp_path / "runbooks" / "anchor_only_7b.yaml",
            "infer_batch_20b_with_notify.anchor_plus_template": tmp_path / "runbooks" / "anchor_plus_template_20b.yaml",
            "infer_batch_20b_with_notify.anchor_only": tmp_path / "runbooks" / "anchor_only_20b.yaml",
            "densegen_batch_with_notify": tmp_path / "runbooks" / "densegen.yaml",
        },
        missing_execution_surfaces=(),
        phase_states=(
            {
                "id": "infer_anchor_only_20b",
                "status": "planned",
                "next_surface": "runbooks/anchor_only_20b.yaml",
            },
            {
                "id": "infer_anchor_plus_template_20b",
                "status": "planned",
                "next_surface": "runbooks/anchor_plus_template_20b.yaml",
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
        evidence={},
    )

    def _resolve_named_path_mapping(value, *, repo_root, label, status_kind):
        del repo_root, label, status_kind
        return {name: Path(path) for name, path in dict(value or {}).items()}

    def _resolve_infer_runtime_lane_contracts(config_paths, *, preferred_model_family):
        del preferred_model_family
        return (
            SimpleNamespace(
                phase_id="infer_anchor_only_20b",
                config_label="anchor_only_20b",
                runtime_label="anchor_only_20b",
                config_path=config_paths["anchor_only_20b"],
            ),
            SimpleNamespace(
                phase_id="infer_anchor_plus_template_20b",
                config_label="anchor_plus_template_20b",
                runtime_label="anchor_plus_template_20b",
                config_path=config_paths["anchor_plus_template_20b"],
            ),
            SimpleNamespace(
                phase_id="infer_anchor_only_7b",
                config_label="anchor_only_7b",
                runtime_label="anchor_only_7b",
                config_path=config_paths["anchor_only_7b"],
            ),
        )

    resolved = resolve_promoter_preflight_context(
        study_context=study_context,
        scope="full",
        status_kind="promoter-study-preflight",
        contract=StudyOpsContract(
            study_id="demo_study",
            family="promoter",
            phase_order=(
                "densegen_growth",
                "construct_context_expansion",
                "infer_batch_preparation",
                "infer_anchor_only_20b",
                "infer_anchor_plus_template_20b",
                "infer_anchor_only_7b",
            ),
            snapshot_summary_scope="repo",
            preflight=StudyPreflightContract(
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
                        "infer_batch_preparation": (
                            "infer",
                            "notify_environment",
                            "notify",
                            "infer_batch_plan",
                        ),
                    },
                    runtime_phase_groups=("infer", "notify", "infer_batch_plan"),
                    runtime_shared_groups=("notify_environment",),
                ),
            ),
            current_phase_id="infer_batch_preparation",
            phases=(
                StudyPhaseContract(id="densegen_growth", status="parallel_optional"),
                StudyPhaseContract(id="construct_context_expansion", status="complete"),
                StudyPhaseContract(id="infer_batch_preparation", status="in_progress"),
                StudyPhaseContract(
                    id="infer_anchor_only_20b",
                    status="planned",
                    next_surface="runbooks/anchor_only_20b.yaml",
                ),
                StudyPhaseContract(
                    id="infer_anchor_plus_template_20b",
                    status="planned",
                    next_surface="runbooks/anchor_plus_template_20b.yaml",
                ),
                StudyPhaseContract(
                    id="infer_anchor_only_7b",
                    status="planned",
                    next_surface="runbooks/anchor_only_7b.yaml",
                ),
            ),
            raw_payload={},
        ),
        dependencies=PromoterPreflightContextDependencies(
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
            resolve_notify_environment_state=lambda *, environ: {
                "NOTIFY_WEBHOOK": bool(environ.get("NOTIFY_WEBHOOK")),
                "NOTIFY_WEBHOOK_FILE": bool(environ.get("NOTIFY_WEBHOOK_FILE")),
                "SSL_CERT_FILE": bool(environ.get("SSL_CERT_FILE")),
            },
            environ={},
        ),
    )

    assert [target.check_id for target in resolved.infer_batch_targets] == [
        "ops.runbook_plan.infer_batch_20b_with_notify.anchor_only",
        "ops.runbook_plan.infer_batch_20b_with_notify.anchor_plus_template",
        "ops.runbook_plan.infer_batch_7b_with_notify.anchor_only",
    ]
    assert [target.phase_id for target in resolved.infer_batch_targets] == [
        "infer_anchor_only_20b",
        "infer_anchor_plus_template_20b",
        "infer_anchor_only_7b",
    ]
    assert [str(target.runbook_path.name) for target in resolved.infer_batch_targets] == [
        "anchor_only_20b.yaml",
        "anchor_plus_template_20b.yaml",
        "anchor_only_7b.yaml",
    ]
