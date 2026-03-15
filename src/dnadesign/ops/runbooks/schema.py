"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/runbooks/schema.py

Strict schema and loader for top-level orchestration runbooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .runbook_layout import enforce_workspace_layout
from .runbook_paths import DENSEGEN_POST_RUN_TEMPLATE_DEFAULT, resolve_runbook_paths
from .workflow_metadata import (
    NotifyPolicy,
    OrchestrationWorkflowId,
    WorkflowTool,
    is_densegen_workflow_id,
    is_infer_workflow_id,
    list_workflow_tools,
    resolve_workflow_tool,
    validate_workflow_contract,
)

_HRT_PATTERN = re.compile(r"^[0-9]{2}:[0-9]{2}:[0-9]{2}$")
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_USR_OVERLAY_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class DensegenRunArgs(StrictBaseModel):
    fresh: str = "--fresh --no-plot"
    resume: str = "--resume --no-plot"

    @field_validator("fresh", "resume")
    @classmethod
    def _validate_args(cls, value: str, info: ValidationInfo) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("run args must be non-empty")
        try:
            args = shlex.split(text)
        except ValueError as exc:
            raise ValueError("run args must be shell-parseable") from exc
        required_flag = "--fresh" if info.field_name == "fresh" else "--resume"
        if required_flag not in args:
            raise ValueError(f"run args for {info.field_name} mode must include {required_flag}")
        return text


class CpuResourceContract(StrictBaseModel):
    pe_omp: int = Field(ge=1)
    h_rt: str
    mem_per_core: str

    @field_validator("h_rt")
    @classmethod
    def _validate_h_rt(cls, value: str) -> str:
        text = str(value).strip()
        if not _HRT_PATTERN.match(text):
            raise ValueError("resources.h_rt must match HH:MM:SS")
        return text

    @field_validator("mem_per_core")
    @classmethod
    def _validate_mem_per_core(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("resources.mem_per_core must be non-empty")
        return text


def _default_densegen_post_run_resources() -> CpuResourceContract:
    return CpuResourceContract(pe_omp=4, h_rt="01:00:00", mem_per_core="4G")


class DensegenPostRunContract(StrictBaseModel):
    qsub_template: Path = DENSEGEN_POST_RUN_TEMPLATE_DEFAULT
    resources: CpuResourceContract = Field(default_factory=_default_densegen_post_run_resources)

    @field_validator("qsub_template")
    @classmethod
    def _normalize_path(cls, value: Path) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("path must be non-empty")
        return Path(text).expanduser()


class UsrOverlayGuardContract(StrictBaseModel):
    max_projected_overlay_parts: int = Field(default=10_000, ge=1)
    max_existing_overlay_parts: int = Field(default=1_000, ge=1)
    auto_compact_existing_overlay_parts: bool = True
    overlay_namespace: str = "densegen"

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_namespace_key(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if "namespace" in value:
            raise ValueError("overlay_guard.namespace is not supported; use overlay_guard.overlay_namespace")
        return value

    @field_validator("overlay_namespace")
    @classmethod
    def _validate_overlay_namespace(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("densegen.overlay_guard.overlay_namespace must be non-empty")
        if _USR_OVERLAY_NAMESPACE_PATTERN.fullmatch(text) is None:
            raise ValueError(
                "densegen.overlay_guard.overlay_namespace must match ^[a-z][a-z0-9_]*$ "
                "for USR overlay namespace compatibility"
            )
        return text


class DensegenRecordsPartGuardContract(StrictBaseModel):
    max_projected_records_parts: int = Field(default=10_000, ge=1)
    max_existing_records_parts: int = Field(default=1_000, ge=1)
    max_existing_records_part_age_days: int = Field(default=14, ge=1)
    auto_compact_existing_records_parts: bool = True


class UsrArchivedOverlayGuardContract(StrictBaseModel):
    max_archived_entries: int = Field(default=1_000, ge=0)
    max_archived_bytes: int = Field(default=2_147_483_648, ge=0)


class DensegenWorkloadContract(StrictBaseModel):
    config: Path
    qsub_template: Path = Path("docs/bu-scc/jobs/densegen-cpu.qsub")
    run_args: DensegenRunArgs = Field(default_factory=DensegenRunArgs)
    post_run: DensegenPostRunContract = Field(default_factory=DensegenPostRunContract)
    overlay_guard: UsrOverlayGuardContract = Field(default_factory=UsrOverlayGuardContract)
    records_part_guard: DensegenRecordsPartGuardContract = Field(default_factory=DensegenRecordsPartGuardContract)
    archived_overlay_guard: UsrArchivedOverlayGuardContract = Field(default_factory=UsrArchivedOverlayGuardContract)

    @field_validator("config", "qsub_template")
    @classmethod
    def _normalize_path(cls, value: Path) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("path must be non-empty")
        return Path(text).expanduser()


def _default_infer_overlay_guard() -> UsrOverlayGuardContract:
    return UsrOverlayGuardContract(overlay_namespace="infer")


class InferWorkloadContract(StrictBaseModel):
    config: Path
    qsub_template: Path = Path("docs/bu-scc/jobs/evo2-gpu-infer.qsub")
    cuda_module: str
    gcc_module: str
    overlay_guard: UsrOverlayGuardContract = Field(default_factory=_default_infer_overlay_guard)

    @field_validator("config", "qsub_template")
    @classmethod
    def _normalize_path(cls, value: Path) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("path must be non-empty")
        return Path(text).expanduser()

    @field_validator("cuda_module", "gcc_module")
    @classmethod
    def _validate_module_name(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("module name must be non-empty")
        return text


class NotifyContract(StrictBaseModel):
    tool: WorkflowTool
    policy: NotifyPolicy = "generic"
    profile: Path
    cursor: Path
    spool_dir: Path
    webhook_env: str = "NOTIFY_WEBHOOK"
    orchestration_events: bool = True
    qsub_template: Path = Path("docs/bu-scc/jobs/notify-watch.qsub")
    smoke: Literal["dry", "live"] = "dry"

    @field_validator("profile", "cursor", "spool_dir", "qsub_template")
    @classmethod
    def _normalize_path(cls, value: Path) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("path must be non-empty")
        return Path(text).expanduser()

    @field_validator("webhook_env")
    @classmethod
    def _validate_webhook_env(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("notify.webhook_env must be non-empty")
        return text


class ResourceContract(CpuResourceContract):
    gpus: int | None = Field(default=None, ge=1)
    gpu_capability: str | None = None
    gpu_memory_gib: float | None = Field(default=None, gt=0)

    @field_validator("gpu_capability")
    @classmethod
    def _validate_gpu_capability(cls, value: str | None) -> str | None:
        if value is None:
            return value
        text = str(value).strip()
        if not text:
            raise ValueError("resources.gpu_capability must be non-empty when provided")
        return text

    @model_validator(mode="after")
    def _validate_gpu_memory_contract(self) -> "ResourceContract":
        if self.gpu_memory_gib is not None and self.gpus is None:
            raise ValueError("resources.gpu_memory_gib requires resources.gpus")
        return self


class ModePolicy(StrictBaseModel):
    default: Literal["auto", "fresh", "resume"] = "auto"
    on_active_job: Literal["hold_jid", "stop"] = "hold_jid"


class LogRetentionContract(StrictBaseModel):
    keep_last: int = Field(default=20, ge=1)
    max_age_days: int = Field(default=14, ge=1)


class LoggingContract(StrictBaseModel):
    stdout_dir: Path
    retention: LogRetentionContract = Field(default_factory=LogRetentionContract)

    @field_validator("stdout_dir")
    @classmethod
    def _normalize_stdout_dir(cls, value: Path) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("logging.stdout_dir must be non-empty")
        return Path(text).expanduser()


class OrchestrationRunbookV1(StrictBaseModel):
    schema_version: Literal[1] = 1
    id: str
    workflow_id: OrchestrationWorkflowId
    project: str
    workspace_root: Path
    densegen: DensegenWorkloadContract | None = None
    infer: InferWorkloadContract | None = None
    notify: NotifyContract | None = None
    logging: LoggingContract
    resources: ResourceContract
    mode_policy: ModePolicy = Field(default_factory=ModePolicy)

    @field_validator("id", "project")
    @classmethod
    def _validate_slug(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("field must be non-empty")
        if not _SLUG_PATTERN.match(text):
            raise ValueError("value must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
        return text

    @field_validator("workspace_root")
    @classmethod
    def _normalize_workspace_root(cls, value: Path) -> Path:
        text = str(value).strip()
        if not text:
            raise ValueError("workspace_root must be non-empty")
        return Path(text).expanduser()

    @model_validator(mode="after")
    def _validate_workflow_contracts(self) -> "OrchestrationRunbookV1":
        validate_workflow_contract(
            workflow_id=self.workflow_id,
            densegen_present=self.densegen is not None,
            infer_present=self.infer is not None,
            notify_present=self.notify is not None,
            notify_tool=(self.notify.tool if self.notify is not None else None),
            notify_policy=(self.notify.policy if self.notify is not None else None),
            has_gpus=self.resources.gpus is not None,
            has_gpu_capability=self.resources.gpu_capability is not None,
            has_gpu_memory=self.resources.gpu_memory_gib is not None,
        )
        return self


class OrchestrationRunbookRoot(StrictBaseModel):
    runbook: OrchestrationRunbookV1

def load_orchestration_runbook(path: Path, *, raw: dict | None = None) -> OrchestrationRunbookV1:
    runbook_path = path.expanduser().resolve()
    if raw is None:
        if not runbook_path.exists():
            raise FileNotFoundError(f"Runbook not found: {runbook_path}")
        raw = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "runbook" not in raw:
        raise ValueError("orchestration runbook schema v1 requires root key: runbook")
    runbook = OrchestrationRunbookRoot.model_validate(raw).runbook
    resolved_runbook = resolve_runbook_paths(runbook, runbook_base_dir=runbook_path.parent)
    return enforce_workspace_layout(resolved_runbook)
