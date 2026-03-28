"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/promoter/adapter.py

Promoter study-family adapter for OPS status and preflight surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.catalog import discover_repo_root
from dnadesign.ops.preflight import (
    choose_command_summary,
    execute_runbook_plan,
    run_preflight_command,
    safe_json_loads,
)
from dnadesign.ops.status.artifacts import load_yaml_mapping, parquet_row_count
from dnadesign.ops.status.parsing import (
    optional_positive_int,
    required_metadata_text,
    string_list_or_empty,
    string_or_none,
)
from dnadesign.ops.status.paths import (
    required_path,
    resolve_named_path_mapping,
    resolve_repo_relative_path,
)
from dnadesign.studies.core.models import StudyFamilyAdapter, StudyStatusContext
from dnadesign.studies.core.record_locator import discover_active_study_selection

from .infer_runtime import PromoterStudyInferRuntimeDependencies
from .preflight import (
    PromoterPreflightContextDependencies,
    PromoterPreflightCoordinatorDependencies,
    build_promoter_preflight_progress,
    resolve_promoter_preflight_context,
)
from .record_normalizer import PromoterStudyContextDependencies, PromoterStudyResolvedContext
from .record_normalizer import resolve_promoter_study_context as resolve_checked_in_promoter_study_context
from .snapshot import (
    PromoterStudyStatusDependencies,
    build_promoter_study_status,
    resolve_promoter_study_status_context,
)


@dataclass(frozen=True)
class PromoterFamilyContext:
    study_context: PromoterStudyResolvedContext


class PromoterStudyFamilyAdapter(StudyFamilyAdapter):
    family_id = "promoter"

    def load_context(self, *, repo_root: Path | None, study_root: Path | None) -> StudyStatusContext:
        study_context = resolve_checked_in_promoter_study_context(
            study_root,
            repo_root=repo_root,
            status_kind="promoter-study-status",
            dependencies=PromoterStudyContextDependencies(
                discover_active_study_dir=discover_active_study_dir,
                required_path=required_path,
                discover_repo_root=discover_repo_root,
                load_yaml_mapping=load_yaml_mapping,
                resolve_repo_relative_path=resolve_repo_relative_path,
                resolve_named_path_mapping=resolve_named_path_mapping,
                string_or_none=string_or_none,
                optional_positive_int=optional_positive_int,
                required_metadata_text=required_metadata_text,
                parquet_row_count=parquet_row_count,
            ),
        )
        contract = study_context.ops_contract
        if contract is None:
            raise ValueError(
                f"study record missing ops.study.yaml: {study_context.resolved_study_dir / 'ops.study.yaml'}"
            )
        if contract.family != self.family_id:
            raise ValueError(
                f"ops.study.yaml family mismatch for {study_context.resolved_study_dir}: "
                f"expected {self.family_id}, found {contract.family}"
            )
        if contract.study_id != study_context.study_id:
            raise ValueError(
                f"ops.study.yaml study_id mismatch for {study_context.resolved_study_dir}: "
                f"expected {study_context.study_id}, found {contract.study_id}"
            )
        if study_context.study_repo_root is None:
            raise ValueError("promoter study context requires a resolved study_repo_root")
        return StudyStatusContext(
            repo_root=study_context.study_repo_root,
            study_root=study_context.resolved_study_dir,
            contract=contract,
            family_context=PromoterFamilyContext(study_context=study_context),
        )

    def build_snapshot(self, context: StudyStatusContext) -> tuple[str, str, dict[str, object]]:
        study_context = _study_family_context(context).study_context
        evidence = dict(study_context.evidence)
        evidence["ops_study_contract"] = dict(context.contract.raw_payload)
        missing_result = _missing_promoter_study_result(context=study_context, evidence=evidence)
        if missing_result is not None:
            return missing_result

        status_dependencies = PromoterStudyStatusDependencies(
            infer_runtime=build_promoter_study_infer_runtime_dependencies(),
            phase_matches_infer_model_family=phase_matches_infer_model_family,
        )
        status_context = resolve_promoter_study_status_context(
            study_context=study_context,
            status_kind="promoter-study-status",
            dependencies=status_dependencies,
        )
        return build_promoter_study_status(
            study_context=study_context,
            status_context=status_context,
            dependencies=status_dependencies,
            summary_scope=context.contract.snapshot_summary_scope,
        )

    def build_preflight(
        self,
        context: StudyStatusContext,
        *,
        scope: str | None,
    ) -> tuple[str, str, dict[str, object]]:
        study_context = _study_family_context(context).study_context
        evidence = dict(study_context.evidence)
        evidence["ops_study_contract"] = dict(context.contract.raw_payload)
        missing_result = _missing_promoter_study_result(context=study_context, evidence=evidence)
        if missing_result is not None:
            return missing_result

        resolved_context = resolve_promoter_preflight_context(
            study_context=study_context,
            scope=scope,
            status_kind="promoter-study-preflight",
            contract=context.contract,
            dependencies=PromoterPreflightContextDependencies(
                infer_runtime=build_promoter_study_infer_runtime_dependencies(),
                environ=os.environ,
            ),
        )

        return build_promoter_preflight_progress(
            context=resolved_context,
            evidence=evidence,
            dependencies=PromoterPreflightCoordinatorDependencies(
                run_preflight_command=run_preflight_command,
                execute_runbook_plan=execute_runbook_plan,
                safe_json_loads=safe_json_loads,
                choose_command_summary=choose_command_summary,
                inspect_local_gpu_inventory=inspect_local_infer_gpu_inventory,
                environ=os.environ,
            ),
        )


STUDY_FAMILY_ADAPTER = PromoterStudyFamilyAdapter()


def discover_active_study_dir(
    *,
    repo_root: Path | None,
    status_kind: str = "promoter-study-status",
) -> tuple[Path, Path, str]:
    selection = discover_active_study_selection(
        repo_root=repo_root,
        status_kind=status_kind,
    )
    return selection.study_root, selection.index_path, selection.active_study_id


def build_promoter_study_infer_runtime_dependencies() -> PromoterStudyInferRuntimeDependencies:
    from dnadesign.infer import resolve_infer_runtime_lane_contracts

    return PromoterStudyInferRuntimeDependencies(
        resolve_named_path_mapping=resolve_named_path_mapping,
        resolve_infer_runtime_lane_contracts=resolve_infer_runtime_lane_contracts,
        derive_infer_notify_profile_paths=derive_infer_notify_profile_paths,
        load_infer_model_summary=load_infer_model_summary,
        string_or_none=string_or_none,
        string_list_or_empty=string_list_or_empty,
    )


def inspect_local_infer_gpu_inventory() -> dict[str, object]:
    try:
        from dnadesign.infer import inspect_local_gpu_inventory

        payload = inspect_local_gpu_inventory()
    except Exception as exc:
        return {"count": 0, "devices": [], "probe_error": str(exc)}
    if not isinstance(payload, dict):
        return {"count": 0, "devices": [], "probe_error": "infer.inspect_local_gpu_inventory returned invalid data"}
    devices = payload.get("devices")
    resolved_devices = list(devices) if isinstance(devices, list) else []
    return {
        "count": int(payload.get("count") or len(resolved_devices)),
        "devices": resolved_devices,
        "probe_error": string_or_none(payload.get("probe_error")),
    }


def derive_infer_notify_profile_paths(
    infer_config_paths: Mapping[str, Path],
) -> tuple[dict[str, Path], dict[str, str]]:
    if not infer_config_paths:
        return {}, {}
    from dnadesign.infer.contracts import resolve_infer_notify_profile_path

    derived_paths: dict[str, Path] = {}
    derivation_errors: dict[str, str] = {}
    for config_label, config_path in infer_config_paths.items():
        try:
            derived_paths[config_label] = resolve_infer_notify_profile_path(config_path)
        except Exception as exc:
            derivation_errors[config_label] = str(exc)
    return derived_paths, derivation_errors


def load_infer_model_summary(config_path: Path) -> dict[str, object]:
    payload = load_yaml_mapping(config_path, label="infer config")
    model_payload = payload.get("model") or {}
    if not isinstance(model_payload, dict):
        raise ValueError(f"infer config must define a model mapping: {config_path}")
    return {
        "model_id": string_or_none(model_payload.get("id")),
        "device": string_or_none(model_payload.get("device")) or "unknown",
    }


def phase_matches_infer_model_family(*, phase_id: str, model_family: str | None) -> bool:
    from dnadesign.infer import infer_model_family_suffix

    suffix = infer_model_family_suffix(model_family)
    return suffix is not None and suffix in phase_id


def _missing_promoter_study_result(
    *,
    context: PromoterStudyResolvedContext,
    evidence: dict[str, object],
) -> tuple[str, str, dict[str, object]] | None:
    if not context.study_dir_exists:
        return ("missing", "promoter study directory not found", evidence)
    missing_required_files = list(context.missing_required_files)
    missing_declared_present = list(context.missing_declared_present)
    missing_execution_surfaces = list(context.missing_execution_surfaces)
    if missing_required_files:
        summary = "study record missing required files: " + ", ".join(missing_required_files)
        return ("missing", summary, evidence)
    if missing_declared_present:
        summary = "study record references declared-present datasets that are missing: " + ", ".join(
            missing_declared_present
        )
        return ("missing", summary, evidence)
    if missing_execution_surfaces:
        summary = "study record references missing execution surfaces: " + ", ".join(missing_execution_surfaces)
        return ("missing", summary, evidence)
    return None


def _study_family_context(context: StudyStatusContext) -> PromoterFamilyContext:
    if not isinstance(context.family_context, PromoterFamilyContext):
        raise ValueError("promoter status context has invalid family_context payload")
    return context.family_context


__all__ = ["PromoterStudyFamilyAdapter", "STUDY_FAMILY_ADAPTER"]
