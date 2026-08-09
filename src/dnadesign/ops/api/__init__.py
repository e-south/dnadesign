"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/api/__init__.py

Public maintainer-facing Python service entrypoints for Ops.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "ActiveJobProbeError",
    "ActiveJobResolution",
    "ActiveJobResolutionState",
    "BatchExecutionResult",
    "BatchPlan",
    "CampaignScaffold",
    "CampaignStatus",
    "CatalogQuery",
    "InferFillLane",
    "InferFillPlan",
    "OpsJobIdentity",
    "OrchestrationRunbookV1",
    "ProcedureStatus",
    "RunbookCatalog",
    "RuntimeVisibility",
    "SchedulerProbeState",
    "StatusKindSpec",
    "build_batch_plan",
    "build_campaign_scaffold",
    "build_infer_fill_plan",
    "build_procedure_status",
    "build_status_inputs",
    "discover_active_job_ids_for_runbook",
    "discover_infer_runbook_paths_for_study",
    "execute_batch_plan",
    "execute_infer_fill_plan",
    "execute_runbook_plan",
    "filter_runbook_catalog",
    "list_status_kind_specs",
    "load_campaign_status",
    "load_catalog_procedure_details",
    "load_catalog_related_registry_ids",
    "load_orchestration_runbook",
    "load_runbook_catalog",
    "load_status_kind_spec",
    "probe_active_jobs_for_runbook",
    "resolve_active_job_resolution",
    "resolve_ops_job_identity",
    "run_status_kind",
]

_EXPORT_MODULES = {
    "BatchExecutionResult": "dnadesign.ops.orchestrator.execute",
    "BatchPlan": "dnadesign.ops.orchestrator.plan",
    "CampaignScaffold": "dnadesign.ops.status.models",
    "CampaignStatus": "dnadesign.ops.status.models",
    "CatalogQuery": "dnadesign.ops.catalog",
    "ActiveJobProbeError": "dnadesign.ops.orchestrator.state",
    "ActiveJobResolution": "dnadesign.ops.orchestrator.state",
    "ActiveJobResolutionState": "dnadesign.ops.orchestrator.state",
    "InferFillLane": "dnadesign.ops.orchestrator.infer_fill",
    "InferFillPlan": "dnadesign.ops.orchestrator.infer_fill",
    "OpsJobIdentity": "dnadesign.ops.orchestrator.state",
    "OrchestrationRunbookV1": "dnadesign.ops.runbooks.schema",
    "ProcedureStatus": "dnadesign.ops.status.models",
    "RuntimeVisibility": "dnadesign.ops.orchestrator.state",
    "RunbookCatalog": "dnadesign.ops.catalog",
    "SchedulerProbeState": "dnadesign.ops.orchestrator.state",
    "StatusKindSpec": "dnadesign.ops.status.models",
    "build_batch_plan": "dnadesign.ops.orchestrator.plan",
    "build_infer_fill_plan": "dnadesign.ops.orchestrator.infer_fill",
    "build_campaign_scaffold": "dnadesign.ops.status.campaign",
    "build_procedure_status": "dnadesign.ops.status.campaign",
    "build_status_inputs": "dnadesign.ops.status.service",
    "discover_active_job_ids_for_runbook": "dnadesign.ops.orchestrator.state",
    "discover_infer_runbook_paths_for_study": "dnadesign.ops.orchestrator.infer_fill",
    "execute_batch_plan": "dnadesign.ops.orchestrator.execute",
    "execute_infer_fill_plan": "dnadesign.ops.orchestrator.infer_fill",
    "execute_runbook_plan": "dnadesign.ops.preflight.support",
    "filter_runbook_catalog": "dnadesign.ops.catalog",
    "list_status_kind_specs": "dnadesign.ops.status.registry_loader",
    "load_campaign_status": "dnadesign.ops.status.campaign",
    "load_catalog_procedure_details": "dnadesign.ops.catalog",
    "load_catalog_related_registry_ids": "dnadesign.ops.catalog",
    "load_orchestration_runbook": "dnadesign.ops.runbooks.schema",
    "load_runbook_catalog": "dnadesign.ops.catalog",
    "load_status_kind_spec": "dnadesign.ops.status.registry_loader",
    "probe_active_jobs_for_runbook": "dnadesign.ops.orchestrator.state",
    "resolve_active_job_resolution": "dnadesign.ops.orchestrator.state",
    "resolve_ops_job_identity": "dnadesign.ops.orchestrator.state",
    "run_status_kind": "dnadesign.ops.status.service",
}


if TYPE_CHECKING:
    from dnadesign.ops.catalog import (
        CatalogQuery,
        RunbookCatalog,
        filter_runbook_catalog,
        load_catalog_procedure_details,
        load_catalog_related_registry_ids,
        load_runbook_catalog,
    )
    from dnadesign.ops.orchestrator.execute import BatchExecutionResult, execute_batch_plan
    from dnadesign.ops.orchestrator.infer_fill import (
        InferFillLane,
        InferFillPlan,
        build_infer_fill_plan,
        discover_infer_runbook_paths_for_study,
        execute_infer_fill_plan,
    )
    from dnadesign.ops.orchestrator.plan import BatchPlan, build_batch_plan
    from dnadesign.ops.orchestrator.state import (
        ActiveJobProbeError,
        ActiveJobResolution,
        ActiveJobResolutionState,
        OpsJobIdentity,
        RuntimeVisibility,
        SchedulerProbeState,
        discover_active_job_ids_for_runbook,
        probe_active_jobs_for_runbook,
        resolve_active_job_resolution,
        resolve_ops_job_identity,
    )
    from dnadesign.ops.preflight.support import execute_runbook_plan
    from dnadesign.ops.runbooks.schema import OrchestrationRunbookV1, load_orchestration_runbook
    from dnadesign.ops.status.campaign import (
        build_campaign_scaffold,
        build_procedure_status,
        load_campaign_status,
    )
    from dnadesign.ops.status.models import CampaignScaffold, CampaignStatus, ProcedureStatus, StatusKindSpec
    from dnadesign.ops.status.registry_loader import list_status_kind_specs, load_status_kind_spec
    from dnadesign.ops.status.service import build_status_inputs, run_status_kind


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
