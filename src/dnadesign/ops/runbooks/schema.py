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

_HRT_PATTERN = re.compile(r"^[0-9]{2}:[0-9]{2}:[0-9]{2}$")
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_USR_OVERLAY_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_DENSEGEN_POST_RUN_TEMPLATE_DEFAULT = Path("docs/bu-scc/jobs/densegen-analysis.qsub")
_DENSEGEN_WORKFLOW_IDS = frozenset(
    {
        "densegen_batch_submit",
        "densegen_batch_with_notify_slack",
    }
)
_INFER_WORKFLOW_IDS = frozenset(
    {
        "infer_batch_submit",
        "infer_batch_with_notify_slack",
    }
)


def is_densegen_workflow_id(workflow_id: str) -> bool:
    return str(workflow_id or "").strip() in _DENSEGEN_WORKFLOW_IDS


def is_infer_workflow_id(workflow_id: str) -> bool:
    return str(workflow_id or "").strip() in _INFER_WORKFLOW_IDS


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


class DensegenPostRunContract(StrictBaseModel):
    qsub_template: Path = _DENSEGEN_POST_RUN_TEMPLATE_DEFAULT

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
    tool: Literal["densegen", "infer"]
    policy: Literal["densegen", "infer_evo2", "generic"] = "densegen"
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


class ResourceContract(StrictBaseModel):
    pe_omp: int = Field(ge=1)
    h_rt: str
    mem_per_core: str
    gpus: int | None = Field(default=None, ge=1)
    gpu_capability: str | None = None

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

    @field_validator("gpu_capability")
    @classmethod
    def _validate_gpu_capability(cls, value: str | None) -> str | None:
        if value is None:
            return value
        text = str(value).strip()
        if not text:
            raise ValueError("resources.gpu_capability must be non-empty when provided")
        return text


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
    workflow_id: Literal[
        "densegen_batch_submit",
        "densegen_batch_with_notify_slack",
        "infer_batch_submit",
        "infer_batch_with_notify_slack",
    ]
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
        is_densegen_workflow = is_densegen_workflow_id(self.workflow_id)
        is_infer_workflow = is_infer_workflow_id(self.workflow_id)
        expects_notify = self.workflow_id.endswith("_with_notify_slack")

        if is_densegen_workflow:
            if self.densegen is None:
                raise ValueError("densegen workflow requires runbook.densegen block")
            if self.infer is not None:
                raise ValueError("densegen workflow does not accept runbook.infer block")
            if self.resources.gpus is not None or self.resources.gpu_capability is not None:
                raise ValueError("densegen workflow does not accept resources.gpus or resources.gpu_capability")
            if expects_notify:
                if self.notify is None:
                    raise ValueError("densegen notify workflow requires runbook.notify block")
                if self.notify.tool != "densegen":
                    raise ValueError("densegen workflow requires notify.tool=densegen")
                if self.notify.policy == "infer_evo2":
                    raise ValueError("densegen workflow does not accept notify.policy=infer_evo2")
            elif self.notify is not None:
                raise ValueError("densegen_batch_submit does not accept runbook.notify block")

        if is_infer_workflow:
            if self.infer is None:
                raise ValueError("infer workflow requires runbook.infer block")
            if self.densegen is not None:
                raise ValueError("infer workflow does not accept runbook.densegen block")
            if self.resources.gpus is None:
                raise ValueError("infer workflow requires resources.gpus")
            if self.resources.gpu_capability is None:
                raise ValueError("infer workflow requires resources.gpu_capability")
            if expects_notify:
                if self.notify is None:
                    raise ValueError("infer notify workflow requires runbook.notify block")
                if self.notify.tool != "infer":
                    raise ValueError("infer workflow requires notify.tool=infer")
                if self.notify.policy == "densegen":
                    raise ValueError("infer workflow does not accept notify.policy=densegen")
            elif self.notify is not None:
                raise ValueError("infer_batch_submit does not accept runbook.notify block")
        return self


class OrchestrationRunbookRoot(StrictBaseModel):
    runbook: OrchestrationRunbookV1


def _resolve_path_from_runbook_base(path_value: Path, *, runbook_base_dir: Path) -> Path:
    expanded = path_value.expanduser()
    if expanded.is_absolute():
        return expanded
    return (runbook_base_dir / expanded).resolve()


def _resolve_repo_root_from_module() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _resolve_densegen_post_run_template(path_value: Path, *, runbook_base_dir: Path) -> Path:
    expanded = path_value.expanduser()
    if expanded.is_absolute():
        return expanded
    if expanded == _DENSEGEN_POST_RUN_TEMPLATE_DEFAULT:
        repo_root = _resolve_repo_root_from_module()
        if repo_root is not None:
            return (repo_root / expanded).resolve()
    return (runbook_base_dir / expanded).resolve()


def _resolve_runbook_paths(runbook: OrchestrationRunbookV1, *, runbook_base_dir: Path) -> OrchestrationRunbookV1:
    densegen = runbook.densegen
    if densegen is not None:
        post_run = densegen.post_run.model_copy(
            update={
                "qsub_template": _resolve_densegen_post_run_template(
                    densegen.post_run.qsub_template,
                    runbook_base_dir=runbook_base_dir,
                )
            }
        )
        densegen = densegen.model_copy(
            update={
                "config": _resolve_path_from_runbook_base(densegen.config, runbook_base_dir=runbook_base_dir),
                "qsub_template": _resolve_path_from_runbook_base(
                    densegen.qsub_template, runbook_base_dir=runbook_base_dir
                ),
                "post_run": post_run,
            }
        )

    infer = runbook.infer
    if infer is not None:
        infer = infer.model_copy(
            update={
                "config": _resolve_path_from_runbook_base(infer.config, runbook_base_dir=runbook_base_dir),
                "qsub_template": _resolve_path_from_runbook_base(
                    infer.qsub_template, runbook_base_dir=runbook_base_dir
                ),
            }
        )

    notify = runbook.notify
    if notify is not None:
        notify = notify.model_copy(
            update={
                "profile": _resolve_path_from_runbook_base(notify.profile, runbook_base_dir=runbook_base_dir),
                "cursor": _resolve_path_from_runbook_base(notify.cursor, runbook_base_dir=runbook_base_dir),
                "spool_dir": _resolve_path_from_runbook_base(notify.spool_dir, runbook_base_dir=runbook_base_dir),
                "qsub_template": _resolve_path_from_runbook_base(
                    notify.qsub_template, runbook_base_dir=runbook_base_dir
                ),
            }
        )

    logging = runbook.logging.model_copy(
        update={
            "stdout_dir": _resolve_path_from_runbook_base(runbook.logging.stdout_dir, runbook_base_dir=runbook_base_dir)
        }
    )

    return runbook.model_copy(
        update={
            "workspace_root": _resolve_path_from_runbook_base(
                runbook.workspace_root, runbook_base_dir=runbook_base_dir
            ),
            "densegen": densegen,
            "infer": infer,
            "notify": notify,
            "logging": logging,
        }
    )


def _is_path_within(*, path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _enforce_workspace_layout(runbook: OrchestrationRunbookV1) -> OrchestrationRunbookV1:
    workspace_root = runbook.workspace_root.resolve()
    ops_logs_root = (workspace_root / "outputs" / "logs" / "ops" / "sge").resolve()
    if not _is_path_within(path=runbook.logging.stdout_dir, parent=ops_logs_root):
        raise ValueError(f"logging.stdout_dir must be under {ops_logs_root}")
    expected_stdout_dir = (ops_logs_root / runbook.id).resolve()
    if runbook.logging.stdout_dir.resolve() != expected_stdout_dir:
        raise ValueError(f"logging.stdout_dir must be exactly {expected_stdout_dir}")

    if runbook.densegen is not None:
        expected_config = (workspace_root / "config.yaml").resolve()
        if runbook.densegen.config.resolve() != expected_config:
            raise ValueError(f"densegen.config must be {expected_config}")
    if runbook.infer is not None:
        expected_config = (workspace_root / "config.yaml").resolve()
        if runbook.infer.config.resolve() != expected_config:
            raise ValueError(f"infer.config must be {expected_config}")
    if runbook.notify is not None:
        notify_root = (workspace_root / "outputs" / "notify" / runbook.notify.tool).resolve()
        expected_profile = (notify_root / "profile.json").resolve()
        expected_cursor = (notify_root / "cursor").resolve()
        expected_spool = (notify_root / "spool").resolve()
        if runbook.notify.profile.resolve() != expected_profile:
            raise ValueError(f"notify.profile must be {expected_profile}")
        if runbook.notify.cursor.resolve() != expected_cursor:
            raise ValueError(f"notify.cursor must be {expected_cursor}")
        if runbook.notify.spool_dir.resolve() != expected_spool:
            raise ValueError(f"notify.spool_dir must be {expected_spool}")

    return runbook


def load_orchestration_runbook(path: Path, *, raw: dict | None = None) -> OrchestrationRunbookV1:
    runbook_path = path.expanduser().resolve()
    if raw is None:
        if not runbook_path.exists():
            raise FileNotFoundError(f"Runbook not found: {runbook_path}")
        raw = yaml.safe_load(runbook_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "runbook" not in raw:
        raise ValueError("orchestration runbook schema v1 requires root key: runbook")
    runbook = OrchestrationRunbookRoot.model_validate(raw).runbook
    resolved_runbook = _resolve_runbook_paths(runbook, runbook_base_dir=runbook_path.parent)
    return _enforce_workspace_layout(resolved_runbook)
