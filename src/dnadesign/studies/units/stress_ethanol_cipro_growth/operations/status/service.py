"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/service.py

Status and preflight service for the stress_ethanol_cipro_growth study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dnadesign.ops.catalog import discover_repo_root
from dnadesign.ops.preflight import (
    choose_command_summary,
    execute_runbook_plan,
    run_preflight_command,
    safe_json_loads,
)
from dnadesign.ops.status import (
    load_yaml_mapping,
    optional_positive_int,
    parquet_row_count,
    required_metadata_text,
    required_path,
    resolve_named_path_mapping,
    resolve_repo_relative_path,
    string_or_none,
)
from dnadesign.studies.core.models import StudyStatusContext, StudyStatusService
from dnadesign.studies.core.record_locator import discover_active_study_selection

from .analysis_surfaces import inspect_stress_ethanol_cipro_growth_exploratory_analysis
from .downstream_surfaces import inspect_stress_ethanol_cipro_growth_downstream_surfaces
from .latentdna_readiness import inspect_stress_ethanol_cipro_growth_latentdna_readiness
from .preflight import (
    StressEthanolCiproGrowthPreflightContextDependencies,
    StressEthanolCiproGrowthPreflightCoordinatorDependencies,
    build_stress_ethanol_cipro_growth_preflight_progress,
    resolve_stress_ethanol_cipro_growth_preflight_context,
)
from .probes.runtime_dependencies import (
    build_stress_ethanol_cipro_growth_infer_runtime_dependencies,
    inspect_local_infer_gpu_inventory,
    phase_matches_infer_model_family,
)
from .probes.semantic_completeness import inspect_stress_ethanol_cipro_growth_semantic_completeness
from .probes.sequence_view_contracts import inspect_stress_ethanol_cipro_growth_sequence_view_contracts
from .record_normalizer import StressEthanolCiproGrowthContextDependencies, StressEthanolCiproGrowthResolvedContext
from .record_normalizer import (
    resolve_stress_ethanol_cipro_growth_context as resolve_checked_in_stress_ethanol_cipro_growth_context,
)
from .snapshot import (
    StressEthanolCiproGrowthStatusDependencies,
    build_stress_ethanol_cipro_growth_status,
    resolve_stress_ethanol_cipro_growth_status_context,
)


@dataclass(frozen=True)
class StressEthanolCiproGrowthStatusServiceContext:
    study_context: StressEthanolCiproGrowthResolvedContext


class StressEthanolCiproGrowthStatusService(StudyStatusService):
    study_id = "stress_ethanol_cipro_growth"
    status_kind = "stress-ethanol-cipro-growth-status"
    preflight_kind = "stress-ethanol-cipro-growth-preflight"

    def load_context(self, *, repo_root: Path | None, study_root: Path | None) -> StudyStatusContext:
        study_context = resolve_checked_in_stress_ethanol_cipro_growth_context(
            study_root,
            repo_root=repo_root,
            status_kind="stress-ethanol-cipro-growth-status",
            dependencies=StressEthanolCiproGrowthContextDependencies(
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
                "study record missing ops.study.yaml: "
                f"{study_context.resolved_study_dir / 'operations' / 'ops.study.yaml'}"
            )
        if contract.status_kind != self.status_kind:
            raise ValueError(
                f"ops.study.yaml ops_surfaces.status_kind mismatch for {study_context.resolved_study_dir}: "
                f"expected {self.status_kind}, found {contract.status_kind}"
            )
        if contract.preflight_kind != self.preflight_kind:
            raise ValueError(
                f"ops.study.yaml ops_surfaces.preflight_kind mismatch for {study_context.resolved_study_dir}: "
                f"expected {self.preflight_kind}, found {contract.preflight_kind}"
            )
        if contract.study_id != study_context.study_id:
            raise ValueError(
                f"ops.study.yaml study_id mismatch for {study_context.resolved_study_dir}: "
                f"expected {study_context.study_id}, found {contract.study_id}"
            )
        if contract.study_id != self.study_id:
            raise ValueError(
                f"{self.status_kind} only serves study_id {self.study_id!r}; "
                f"found {contract.study_id!r} in {study_context.resolved_study_dir / 'operations' / 'ops.study.yaml'}"
            )
        if study_context.study_repo_root is None:
            raise ValueError("stress_ethanol_cipro_growth context requires a resolved study_repo_root")
        return StudyStatusContext(
            repo_root=study_context.study_repo_root,
            study_root=study_context.resolved_study_dir,
            contract=contract,
            service_context=StressEthanolCiproGrowthStatusServiceContext(study_context=study_context),
        )

    def build_snapshot(self, context: StudyStatusContext) -> tuple[str, str, dict[str, object]]:
        study_context = _study_service_context(context).study_context
        evidence = dict(study_context.evidence)
        evidence["ops_study_contract"] = dict(context.contract.raw_payload)
        missing_result = _missing_stress_ethanol_cipro_growth_result(context=study_context, evidence=evidence)
        if missing_result is not None:
            return missing_result

        status_dependencies = StressEthanolCiproGrowthStatusDependencies(
            infer_runtime=build_stress_ethanol_cipro_growth_infer_runtime_dependencies(),
            phase_matches_infer_model_family=phase_matches_infer_model_family,
            inspect_semantic_completeness=inspect_stress_ethanol_cipro_growth_semantic_completeness,
            inspect_sequence_view_contracts=inspect_stress_ethanol_cipro_growth_sequence_view_contracts,
            inspect_latentdna_readiness=inspect_stress_ethanol_cipro_growth_latentdna_readiness,
            inspect_additional_downstream_surfaces=inspect_stress_ethanol_cipro_growth_downstream_surfaces,
            inspect_exploratory_analysis=inspect_stress_ethanol_cipro_growth_exploratory_analysis,
        )
        status_context = resolve_stress_ethanol_cipro_growth_status_context(
            study_context=study_context,
            status_kind="stress-ethanol-cipro-growth-status",
            dependencies=status_dependencies,
        )
        return build_stress_ethanol_cipro_growth_status(
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
        command_timeout_seconds: object | None = None,
    ) -> tuple[str, str, dict[str, object]]:
        study_context = _study_service_context(context).study_context
        evidence = dict(study_context.evidence)
        evidence["ops_study_contract"] = dict(context.contract.raw_payload)
        resolved_command_timeout_seconds = _resolve_preflight_command_timeout_seconds(command_timeout_seconds)
        if resolved_command_timeout_seconds is not None:
            evidence["command_timeout_seconds"] = resolved_command_timeout_seconds
        missing_result = _missing_stress_ethanol_cipro_growth_result(context=study_context, evidence=evidence)
        if missing_result is not None:
            return missing_result

        resolved_context = resolve_stress_ethanol_cipro_growth_preflight_context(
            study_context=study_context,
            scope=scope,
            status_kind="stress-ethanol-cipro-growth-preflight",
            contract=context.contract,
            dependencies=StressEthanolCiproGrowthPreflightContextDependencies(
                infer_runtime=build_stress_ethanol_cipro_growth_infer_runtime_dependencies(),
                environ=os.environ,
            ),
        )

        return build_stress_ethanol_cipro_growth_preflight_progress(
            context=resolved_context,
            evidence=evidence,
            dependencies=StressEthanolCiproGrowthPreflightCoordinatorDependencies(
                run_preflight_command=_build_preflight_command_runner(resolved_command_timeout_seconds),
                execute_runbook_plan=execute_runbook_plan,
                safe_json_loads=safe_json_loads,
                choose_command_summary=choose_command_summary,
                inspect_local_gpu_inventory=inspect_local_infer_gpu_inventory,
                environ=os.environ,
            ),
        )


STUDY_STATUS_SERVICE = StressEthanolCiproGrowthStatusService()


def _resolve_preflight_command_timeout_seconds(value: object | None) -> int | None:
    resolved = optional_positive_int(value)
    if resolved is None:
        return None
    if resolved <= 0:
        raise ValueError("stress-ethanol-cipro-growth-preflight command_timeout_seconds must be greater than zero")
    return resolved


def _build_preflight_command_runner(command_timeout_seconds: int | None):
    def _run_preflight_command(
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: int | float = 180,
    ):
        return run_preflight_command(
            argv,
            cwd=cwd,
            timeout_seconds=command_timeout_seconds if command_timeout_seconds is not None else timeout_seconds,
        )

    return _run_preflight_command


def discover_active_study_dir(
    *,
    repo_root: Path | None,
    status_kind: str = "stress-ethanol-cipro-growth-status",
) -> tuple[Path, Path, str]:
    selection = discover_active_study_selection(
        repo_root=repo_root,
        status_kind=status_kind,
    )
    return selection.study_root, selection.index_path, selection.active_study_id


def _missing_stress_ethanol_cipro_growth_result(
    *,
    context: StressEthanolCiproGrowthResolvedContext,
    evidence: dict[str, object],
) -> tuple[str, str, dict[str, object]] | None:
    if not context.study_dir_exists:
        return ("missing", "stress_ethanol_cipro_growth study directory not found", evidence)
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


def _study_service_context(context: StudyStatusContext) -> StressEthanolCiproGrowthStatusServiceContext:
    if not isinstance(context.service_context, StressEthanolCiproGrowthStatusServiceContext):
        raise ValueError("stress_ethanol_cipro_growth status context has invalid service_context payload")
    return context.service_context


__all__ = [
    "StressEthanolCiproGrowthStatusService",
    "StressEthanolCiproGrowthStatusServiceContext",
    "STUDY_STATUS_SERVICE",
]
