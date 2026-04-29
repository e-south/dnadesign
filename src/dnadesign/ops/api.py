"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/api.py

Explicit maintainer-facing Python service entrypoints for OPS.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .catalog import (
    CatalogQuery,
    RunbookCatalog,
    filter_runbook_catalog,
    load_catalog_procedure_details,
    load_catalog_related_registry_ids,
    load_runbook_catalog,
)
from .orchestrator.execute import BatchExecutionResult, execute_batch_plan
from .orchestrator.infer_fill import (
    InferFillLane,
    InferFillPlan,
    build_infer_fill_plan,
    discover_infer_runbook_paths_for_study,
    execute_infer_fill_plan,
    resolve_active_study_dir,
)
from .orchestrator.plan import BatchPlan, build_batch_plan
from .orchestrator.state import (
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
from .preflight.support import execute_runbook_plan
from .runbooks.schema import OrchestrationRunbookV1, load_orchestration_runbook
from .status.campaign import build_campaign_scaffold, build_procedure_status, load_campaign_status
from .status.models import CampaignScaffold, CampaignStatus, ProcedureStatus, StatusKindSpec
from .status.registry_loader import list_status_kind_specs, load_status_kind_spec
from .status.service import build_status_inputs, run_status_kind

__all__ = [
    "BatchExecutionResult",
    "BatchPlan",
    "CampaignScaffold",
    "CampaignStatus",
    "CatalogQuery",
    "ActiveJobProbeError",
    "ActiveJobResolution",
    "ActiveJobResolutionState",
    "InferFillLane",
    "InferFillPlan",
    "OpsJobIdentity",
    "OrchestrationRunbookV1",
    "ProcedureStatus",
    "RuntimeVisibility",
    "RunbookCatalog",
    "SchedulerProbeState",
    "StatusKindSpec",
    "build_batch_plan",
    "build_infer_fill_plan",
    "build_campaign_scaffold",
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
    "resolve_active_study_dir",
    "resolve_ops_job_identity",
    "run_status_kind",
]
