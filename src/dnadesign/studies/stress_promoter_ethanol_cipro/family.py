"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/stress_promoter_ethanol_cipro/family.py

Explicit stress_promoter_ethanol_cipro study adapter for OPS status and
preflight surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.catalog import discover_repo_root
from dnadesign.ops.progress_command_support import (
    build_infer_notify_setup_command,
    choose_command_summary,
    infer_usr_dataset_requirements,
    load_orchestration_runbook_payload,
    preflight_command_check,
    preflight_state_check,
    run_progress_command,
    safe_json_loads,
)
from dnadesign.ops.progress_support import (
    load_yaml_mapping,
    optional_positive_int,
    parquet_row_count,
    required_metadata_text,
    required_path,
    resolve_input_path,
    resolve_named_path_mapping,
    resolve_repo_relative_path,
    string_list_or_empty,
    string_or_none,
)
from dnadesign.studies.core.models import StudyFamilyAdapter, StudyStatusContext
from dnadesign.studies.core.record_loader import load_study_ops_contract

from .context import PromoterStudyContextDependencies, PromoterStudyResolvedContext
from .context import resolve_promoter_study_context as resolve_checked_in_promoter_study_context
from .infer_runtime import PromoterStudyInferRuntimeDependencies
from .preflight import (
    PromoterPreflightContextDependencies,
    PromoterPreflightCoordinatorDependencies,
    build_promoter_preflight_progress,
    resolve_promoter_preflight_context,
)
from .preflight_orchestration import resolve_notify_environment_state
from .snapshot import (
    PromoterStudyStatusDependencies,
    build_promoter_study_record_progress,
    resolve_promoter_study_status_context,
)


@dataclass(frozen=True)
class StressPromoterEthanolCiproFamilyContext:
    study_context: PromoterStudyResolvedContext


class StressPromoterEthanolCiproStudyAdapter(StudyFamilyAdapter):
    family_id = "stress_promoter_ethanol_cipro"

    def load_context(self, *, repo_root: Path | None, study_root: Path | None) -> StudyStatusContext:
        study_context = resolve_checked_in_promoter_study_context(
            study_root,
            repo_root=repo_root,
            progress_kind="promoter-study-record",
            dependencies=PromoterStudyContextDependencies(
                discover_active_promoter_study_dir=discover_active_study_dir,
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
        contract = load_study_ops_contract(study_context.resolved_study_dir)
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
            family_context=StressPromoterEthanolCiproFamilyContext(study_context=study_context),
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
            inspect_local_gpu_inventory=inspect_local_infer_gpu_inventory,
            phase_matches_infer_model_family=phase_matches_infer_model_family,
        )
        status_context = resolve_promoter_study_status_context(
            study_context=study_context,
            progress_kind="promoter-study-record",
            dependencies=status_dependencies,
        )
        return build_promoter_study_record_progress(
            study_context=study_context,
            status_context=status_context,
            dependencies=status_dependencies,
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
            progress_kind="promoter-study-preflight",
            contract=context.contract,
            dependencies=PromoterPreflightContextDependencies(
                infer_runtime=build_promoter_study_infer_runtime_dependencies(),
                resolve_notify_environment_state=resolve_notify_environment_state,
                environ=os.environ,
            ),
        )

        validate_infer_config_contract = None
        validate_infer_dry_run_contract = None
        resolve_infer_usr_output_contract = None
        if resolved_context.scope_plan.includes_group("infer") or resolved_context.scope_plan.includes_group("notify"):
            from dnadesign.infer import validate_infer_config_contract, validate_infer_dry_run_contract
            from dnadesign.infer.contracts import resolve_infer_usr_output_contract

        return build_promoter_preflight_progress(
            context=resolved_context,
            evidence=evidence,
            dependencies=PromoterPreflightCoordinatorDependencies(
                load_orchestration_runbook_payload=load_orchestration_runbook_payload,
                resolve_input_path=lambda path, base_dir: resolve_input_path(path, base_dir=base_dir),
                run_progress_command=run_progress_command,
                safe_json_loads=safe_json_loads,
                preflight_state_check=preflight_state_check,
                preflight_command_check=preflight_command_check,
                choose_command_summary=choose_command_summary,
                inspect_local_gpu_inventory=inspect_local_infer_gpu_inventory,
                infer_usr_dataset_requirements=infer_usr_dataset_requirements,
                build_infer_notify_setup_command=lambda config_path: build_infer_notify_setup_command(
                    config_path=config_path
                ),
                validate_infer_config_contract=validate_infer_config_contract,
                validate_infer_dry_run_contract=validate_infer_dry_run_contract,
                resolve_infer_usr_output_contract=resolve_infer_usr_output_contract,
            ),
        )


STRESS_PROMOTER_ETHANOL_CIPRO_STUDY_ADAPTER = StressPromoterEthanolCiproStudyAdapter()


def discover_active_study_dir(
    *,
    repo_root: Path | None,
    progress_kind: str = "promoter-study-record",
) -> tuple[Path, Path, str]:
    resolved_repo_root = repo_root
    if resolved_repo_root is None:
        resolved_repo_root = discover_repo_root(Path.cwd())
    if resolved_repo_root is None:
        raise ValueError(
            f"progress kind '{progress_kind}' requires --study-dir or a dnadesign repository checkout "
            "with docs/studies/promoter/index.yaml"
        )

    study_index_path = resolved_repo_root / "docs" / "studies" / "promoter" / "index.yaml"
    if not study_index_path.exists():
        raise ValueError(f"stress_promoter_ethanol_cipro study registry not found: {study_index_path}")

    study_index = load_yaml_mapping(study_index_path, label="stress_promoter_ethanol_cipro index")
    active_study = string_or_none(study_index.get("active_study"))
    if active_study is None:
        raise ValueError(f"stress_promoter_ethanol_cipro registry does not declare active_study: {study_index_path}")

    studies_payload = study_index.get("studies") or []
    if not isinstance(studies_payload, list):
        raise ValueError(f"stress_promoter_ethanol_cipro registry must define a 'studies' list: {study_index_path}")

    matching_entries = [
        entry
        for entry in studies_payload
        if isinstance(entry, dict) and string_or_none(entry.get("study_id")) == active_study
    ]
    if not matching_entries:
        raise ValueError(f"active_study '{active_study}' is not declared under 'studies' in {study_index_path}")
    if len(matching_entries) > 1:
        raise ValueError(f"active_study '{active_study}' is declared more than once in {study_index_path}")

    raw_path = required_metadata_text(
        matching_entries[0].get("path"),
        label="study path",
        source=study_index_path,
    )
    resolved_study_dir = resolve_repo_relative_path(
        repo_root=resolved_repo_root,
        raw_path=raw_path,
        progress_kind=progress_kind,
    )
    return resolved_study_dir, study_index_path, active_study


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


def _study_family_context(context: StudyStatusContext) -> StressPromoterEthanolCiproFamilyContext:
    if not isinstance(context.family_context, StressPromoterEthanolCiproFamilyContext):
        raise ValueError("stress_promoter_ethanol_cipro status context has invalid family_context payload")
    return context.family_context


__all__ = [
    "STRESS_PROMOTER_ETHANOL_CIPRO_STUDY_ADAPTER",
    "StressPromoterEthanolCiproStudyAdapter",
]
