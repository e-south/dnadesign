"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/state.py

Deterministic run-mode and active-job submission-behavior resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from dnadesign.densegen.contracts import resolve_densegen_usr_output_contract
from dnadesign.ops.contracts import ResumeReadinessPolicy, resolve_resume_readiness_policy

from ..runbooks.schema import OrchestrationRunbookV1
from .mode_tools import InferModeProbeError, resolve_mode_tool_adapter

RunMode = Literal["auto", "fresh", "resume"]
SubmitBehavior = Literal["submit", "hold_jid", "blocked"]
ResumeState = Literal["none", "resume_ready", "partial"]
_OPS_IDENTITY_KEYS = ("ops_run_group_id", "ops_workspace_id", "ops_workflow_id")
_SCHEDULER_PROBE_TIMEOUT_SECONDS = 10.0
_SCHEDULER_DISCOVERY_BUDGET_SECONDS = 10.0
_SUBMIT_HOST_DENIED_TOKENS = (
    "is no submit host",
    "neither submit nor admin host",
)


def _run_probe(
    argv: Sequence[str],
    *,
    timeout_seconds: float | None = None,
) -> tuple[int, str, str]:
    effective_timeout = _SCHEDULER_PROBE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    try:
        result = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
    except OSError as exc:
        cmd = str(argv[0]) if argv else "command"
        return 127, "", f"{cmd} unavailable: {exc}"
    except subprocess.TimeoutExpired:
        cmd = str(argv[0]) if argv else "command"
        return 124, "", f"{cmd} unavailable: timed out after {effective_timeout:g} seconds"
    return int(result.returncode), result.stdout, result.stderr


def _parse_job_ids_from_qstat_output(text: str) -> tuple[str, ...]:
    job_ids: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        job_ids.append(parts[0])
    return tuple(job_ids)


def _slug_token(value: str, *, fallback: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value or "").strip()).strip("._-")
    return token or fallback


def _short_digest(*parts: object, length: int = 12) -> str:
    payload = "\n".join(str(part or "").strip() for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


class SchedulerProbeState(StrEnum):
    SKIPPED = "skipped"
    OK = "ok"
    HOST_DENIED = "host_denied"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ActiveJobResolutionState(StrEnum):
    NOT_REQUIRED = "not_required"
    NO_MATCH = "no_match"
    MATCHED = "matched"
    MULTIPLE_MATCHES = "multiple_matches"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RuntimeVisibility:
    scheduler_probe_state: SchedulerProbeState
    active_job_resolution_state: ActiveJobResolutionState
    degraded: bool
    degraded_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "scheduler_probe_state": self.scheduler_probe_state.value,
            "active_job_resolution_state": self.active_job_resolution_state.value,
            "degraded": self.degraded,
            "degraded_reasons": list(self.degraded_reasons),
        }


@dataclass(frozen=True)
class ActiveJobResolution:
    explicit_job_ids: tuple[str, ...]
    discovered_job_ids: tuple[str, ...]
    effective_job_ids: tuple[str, ...]
    runtime_visibility: RuntimeVisibility

    def as_dict(self) -> dict[str, object]:
        return {
            "explicit_job_ids": list(self.explicit_job_ids),
            "discovered_job_ids": list(self.discovered_job_ids),
            "effective_job_ids": list(self.effective_job_ids),
            "runtime_visibility": self.runtime_visibility.as_dict(),
        }


class ActiveJobProbeError(RuntimeError):
    def __init__(self, runtime_visibility: RuntimeVisibility):
        reasons = runtime_visibility.degraded_reasons or ("active-job visibility is unavailable",)
        super().__init__("; ".join(reasons))
        self.runtime_visibility = runtime_visibility


@dataclass(frozen=True)
class OpsJobIdentity:
    workflow_id: str
    run_group_id: str
    workspace_id: str
    job_name_slug: str
    runbook_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "workflow_id": self.workflow_id,
            "run_group_id": self.run_group_id,
            "workspace_id": self.workspace_id,
            "job_name_slug": self.job_name_slug,
            "runbook_id": self.runbook_id,
        }


def resolve_ops_job_identity(runbook: OrchestrationRunbookV1) -> OpsJobIdentity:
    workspace_root = str(runbook.workspace_root.resolve())
    run_group_id = _short_digest(runbook.workflow_id, runbook.id, workspace_root, length=16)
    workspace_id = _short_digest(workspace_root, length=12)
    job_name_slug = _slug_token(f"{runbook.id}.{run_group_id[:8]}", fallback="ops")
    return OpsJobIdentity(
        workflow_id=runbook.workflow_id,
        run_group_id=run_group_id,
        workspace_id=workspace_id,
        job_name_slug=job_name_slug,
        runbook_id=runbook.id,
    )


def build_ops_job_context(identity: OpsJobIdentity, *, role: str | None = None) -> dict[str, str]:
    context = {
        "ops_job_name_slug": identity.job_name_slug,
        "ops_run_group_id": identity.run_group_id,
        "ops_runbook_id": identity.runbook_id,
        "ops_workflow_id": identity.workflow_id,
        "ops_workspace_id": identity.workspace_id,
    }
    if role is not None:
        context["ops_job_role"] = _slug_token(role, fallback="job")
    return context


def build_ops_job_env(identity: OpsJobIdentity, *, role: str | None = None) -> dict[str, str]:
    context = build_ops_job_context(identity, role=role)
    return {key.upper(): value for key, value in context.items()}


def render_sge_context_value(context: Mapping[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in sorted(context.items()))


def render_sge_job_name(identity: OpsJobIdentity, *, role: str) -> str:
    role_token = _slug_token(role, fallback="job")
    return f"ops.{identity.job_name_slug}.{role_token}"[:128]


def _parse_assignment_list(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in str(value or "").split(","):
        token = item.strip()
        if not token or "=" not in token:
            continue
        key, raw_value = token.split("=", maxsplit=1)
        normalized_key = key.strip().lower()
        normalized_value = raw_value.strip()
        if normalized_key and normalized_value:
            result[normalized_key] = normalized_value
    return result


def _parse_qstat_job_metadata(text: str) -> tuple[str | None, dict[str, str]]:
    job_name: str | None = None
    tags: dict[str, str] = {}
    for line in str(text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key == "job_name":
            job_name = normalized_value or None
            continue
        if normalized_key in {"context", "env_list"}:
            tags.update(_parse_assignment_list(normalized_value))
    return job_name, tags


def _job_name_matches_identity(identity: OpsJobIdentity, job_name: str | None) -> bool:
    if not job_name:
        return False
    expected_prefix = f"ops.{identity.job_name_slug}."
    return str(job_name).startswith(expected_prefix)


def _job_matches_identity(identity: OpsJobIdentity, tags: Mapping[str, str]) -> bool:
    expected = {
        "ops_run_group_id": identity.run_group_id,
        "ops_workspace_id": identity.workspace_id,
        "ops_workflow_id": identity.workflow_id,
    }
    return all(tags.get(key) == value for key, value in expected.items())


def _job_exposes_identity_contract(tags: Mapping[str, str]) -> bool:
    return all(str(tags.get(key) or "").strip() for key in _OPS_IDENTITY_KEYS)


def _active_job_resolution_state_for_job_ids(job_ids: Sequence[str]) -> ActiveJobResolutionState:
    unique_job_ids = tuple(dict.fromkeys(str(job_id).strip() for job_id in job_ids if str(job_id).strip()))
    if not unique_job_ids:
        return ActiveJobResolutionState.NO_MATCH
    if len(unique_job_ids) == 1:
        return ActiveJobResolutionState.MATCHED
    return ActiveJobResolutionState.MULTIPLE_MATCHES


def _dedupe_job_ids(job_ids: Sequence[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for job_id in job_ids:
        for token in str(job_id).split(","):
            normalized = token.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
    return tuple(deduped)


def _default_runtime_visibility_for_job_ids(job_ids: Sequence[str]) -> RuntimeVisibility:
    deduped = _dedupe_job_ids(job_ids)
    if not deduped:
        return RuntimeVisibility(
            scheduler_probe_state=SchedulerProbeState.SKIPPED,
            active_job_resolution_state=ActiveJobResolutionState.NOT_REQUIRED,
            degraded=False,
        )
    return RuntimeVisibility(
        scheduler_probe_state=SchedulerProbeState.SKIPPED,
        active_job_resolution_state=_active_job_resolution_state_for_job_ids(deduped),
        degraded=False,
    )


def _is_submit_host_denied_message(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    return any(token in normalized for token in _SUBMIT_HOST_DENIED_TOKENS)


def default_runtime_visibility() -> RuntimeVisibility:
    return _default_runtime_visibility_for_job_ids(())


def default_runtime_visibility_for_job_ids(job_ids: Sequence[str]) -> RuntimeVisibility:
    return _default_runtime_visibility_for_job_ids(job_ids)


def probe_active_jobs_for_runbook(
    runbook: OrchestrationRunbookV1,
    *,
    max_jobs: int = 24,
    budget_seconds: float = _SCHEDULER_DISCOVERY_BUDGET_SECONDS,
) -> ActiveJobResolution:
    if max_jobs <= 0:
        return ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=RuntimeVisibility(
                scheduler_probe_state=SchedulerProbeState.SKIPPED,
                active_job_resolution_state=ActiveJobResolutionState.NOT_REQUIRED,
                degraded=False,
            ),
        )

    if budget_seconds <= 0:
        return ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=RuntimeVisibility(
                scheduler_probe_state=SchedulerProbeState.BUDGET_EXHAUSTED,
                active_job_resolution_state=ActiveJobResolutionState.UNKNOWN,
                degraded=True,
                degraded_reasons=("active-job discovery requires an overall probe budget greater than zero",),
            ),
        )

    identity = resolve_ops_job_identity(runbook)
    deadline = time.monotonic() + budget_seconds
    user = os.environ.get("USER", "")
    return_code, stdout, stderr = _run_probe(
        ("qstat", "-u", user),
        timeout_seconds=max(0.001, deadline - time.monotonic()),
    )
    if return_code != 0:
        message = stderr.strip() or stdout.strip() or "qstat -u failed"
        probe_state = (
            SchedulerProbeState.BUDGET_EXHAUSTED
            if return_code == 124
            else (
                SchedulerProbeState.HOST_DENIED
                if _is_submit_host_denied_message(message)
                else SchedulerProbeState.UNAVAILABLE
            )
        )
        return ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=RuntimeVisibility(
                scheduler_probe_state=probe_state,
                active_job_resolution_state=ActiveJobResolutionState.UNKNOWN,
                degraded=True,
                degraded_reasons=(message,),
            ),
        )

    active_job_ids: list[str] = []
    degraded_reasons: list[str] = []
    scheduler_probe_state = SchedulerProbeState.OK
    queued_job_ids = _parse_job_ids_from_qstat_output(stdout)
    inspected_jobs = 0
    for job_id in queued_job_ids:
        if inspected_jobs >= max_jobs:
            degraded_reasons.append(
                f"active-job discovery inspected {inspected_jobs} of {len(queued_job_ids)} queued jobs; "
                f"--max-discovery-jobs={max_jobs} exhausted"
            )
            scheduler_probe_state = SchedulerProbeState.BUDGET_EXHAUSTED
            break
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            degraded_reasons.append(
                f"active-job discovery exceeded its {budget_seconds:g} second overall probe budget "
                f"after inspecting {inspected_jobs} of {len(queued_job_ids)} queued jobs"
            )
            scheduler_probe_state = SchedulerProbeState.BUDGET_EXHAUSTED
            break
        inspected_jobs += 1
        rc, job_stdout, job_stderr = _run_probe(
            ("qstat", "-j", str(job_id)),
            timeout_seconds=max(0.001, remaining_seconds),
        )
        if rc != 0:
            message = job_stderr.strip() or job_stdout.strip() or f"qstat -j {job_id} failed"
            degraded_reasons.append(f"active-job detail probe failed for job {job_id}: {message}")
            if rc == 124:
                degraded_reasons.append(
                    f"active-job discovery exceeded its {budget_seconds:g} second overall probe budget "
                    f"after inspecting {inspected_jobs} of {len(queued_job_ids)} queued jobs"
                )
                scheduler_probe_state = SchedulerProbeState.BUDGET_EXHAUSTED
                break
            scheduler_probe_state = SchedulerProbeState.ERROR
            continue
        job_name, tags = _parse_qstat_job_metadata(job_stdout)
        if _job_matches_identity(identity, tags):
            active_job_ids.append(str(job_id))
            continue
        if _job_name_matches_identity(identity, job_name) and not _job_exposes_identity_contract(tags):
            degraded_reasons.append(
                "scheduler surfaced candidate job "
                f"{job_id} without the explicit OPS identity tags required for discovery"
            )
            scheduler_probe_state = SchedulerProbeState.UNSUPPORTED
            continue
    discovered_job_ids = tuple(active_job_ids)
    resolution_state = _active_job_resolution_state_for_job_ids(discovered_job_ids)
    if degraded_reasons:
        resolution_state = ActiveJobResolutionState.UNKNOWN
    return ActiveJobResolution(
        explicit_job_ids=(),
        discovered_job_ids=discovered_job_ids,
        effective_job_ids=discovered_job_ids,
        runtime_visibility=RuntimeVisibility(
            scheduler_probe_state=scheduler_probe_state,
            active_job_resolution_state=resolution_state,
            degraded=bool(degraded_reasons),
            degraded_reasons=tuple(dict.fromkeys(degraded_reasons)),
        ),
    )


def resolve_active_job_resolution(
    *,
    runbook: OrchestrationRunbookV1,
    explicit_job_ids: Sequence[str],
    discover_active_jobs: bool,
    max_jobs: int = 24,
) -> ActiveJobResolution:
    normalized_explicit_job_ids = _dedupe_job_ids(explicit_job_ids)
    if not discover_active_jobs:
        if normalized_explicit_job_ids:
            return ActiveJobResolution(
                explicit_job_ids=normalized_explicit_job_ids,
                discovered_job_ids=(),
                effective_job_ids=normalized_explicit_job_ids,
                runtime_visibility=RuntimeVisibility(
                    scheduler_probe_state=SchedulerProbeState.SKIPPED,
                    active_job_resolution_state=_active_job_resolution_state_for_job_ids(normalized_explicit_job_ids),
                    degraded=False,
                ),
            )
        return ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=RuntimeVisibility(
                scheduler_probe_state=SchedulerProbeState.SKIPPED,
                active_job_resolution_state=ActiveJobResolutionState.UNKNOWN,
                degraded=True,
                degraded_reasons=(
                    f"active-job discovery skipped while mode_policy.on_active_job={runbook.mode_policy.on_active_job}",
                ),
            ),
        )

    try:
        auto_resolution = probe_active_jobs_for_runbook(runbook, max_jobs=max_jobs)
    except RuntimeError as exc:
        probe_state = (
            SchedulerProbeState.HOST_DENIED
            if _is_submit_host_denied_message(str(exc))
            else SchedulerProbeState.UNAVAILABLE
        )
        auto_resolution = ActiveJobResolution(
            explicit_job_ids=(),
            discovered_job_ids=(),
            effective_job_ids=(),
            runtime_visibility=RuntimeVisibility(
                scheduler_probe_state=probe_state,
                active_job_resolution_state=ActiveJobResolutionState.UNKNOWN,
                degraded=True,
                degraded_reasons=(str(exc),),
            ),
        )
    effective_job_ids = _dedupe_job_ids((*normalized_explicit_job_ids, *auto_resolution.discovered_job_ids))
    runtime_visibility = auto_resolution.runtime_visibility
    if normalized_explicit_job_ids:
        runtime_visibility = RuntimeVisibility(
            scheduler_probe_state=auto_resolution.runtime_visibility.scheduler_probe_state,
            active_job_resolution_state=_active_job_resolution_state_for_job_ids(effective_job_ids),
            degraded=auto_resolution.runtime_visibility.degraded,
            degraded_reasons=auto_resolution.runtime_visibility.degraded_reasons,
        )
    return ActiveJobResolution(
        explicit_job_ids=normalized_explicit_job_ids,
        discovered_job_ids=auto_resolution.discovered_job_ids,
        effective_job_ids=effective_job_ids,
        runtime_visibility=runtime_visibility,
    )


def discover_active_job_ids_for_runbook(
    runbook: OrchestrationRunbookV1,
    *,
    max_jobs: int = 24,
) -> tuple[str, ...]:
    resolution = probe_active_jobs_for_runbook(runbook, max_jobs=max_jobs)
    runtime_visibility = resolution.runtime_visibility
    if runtime_visibility.scheduler_probe_state != SchedulerProbeState.OK or runtime_visibility.degraded:
        raise ActiveJobProbeError(runtime_visibility)
    return resolution.discovered_job_ids


def _normalize_hold_jid(active_job_ids: Sequence[str]) -> str | None:
    normalized: list[str] = []
    seen: set[str] = set()
    for job_id in active_job_ids:
        for value in str(job_id).split(","):
            token = value.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            normalized.append(token)
    if not normalized:
        return None
    return ",".join(sorted(normalized))


@dataclass(frozen=True)
class ModeDecision:
    requested_mode: RunMode
    selected_mode: Literal["fresh", "resume"]
    run_args: str
    resume_artifacts_found: bool
    submit_behavior: SubmitBehavior
    hold_jid: str | None
    reason: str


def _densegen_usr_record_candidates(runbook: OrchestrationRunbookV1) -> tuple[Path, ...]:
    if runbook.densegen is None:
        return ()
    if not runbook.densegen.config.exists() or not runbook.densegen.config.is_file():
        return ()
    contract = resolve_densegen_usr_output_contract(runbook.densegen.config)
    dataset_root = contract.usr_root / contract.usr_dataset
    return (dataset_root / "records.parquet",)


def _candidate_record_paths_for_resume(runbook: OrchestrationRunbookV1) -> tuple[Path, ...]:
    workspace_root = runbook.workspace_root
    tables_root = workspace_root / "outputs" / "tables"
    candidate_dirs = [tables_root]
    nested_tables_root = tables_root / "tables"
    if nested_tables_root.exists():
        candidate_dirs.append(nested_tables_root)
    candidates: list[Path] = []
    for directory in candidate_dirs:
        candidates.append(directory / "records.parquet")
        candidates.extend(sorted(directory.glob("records__part-*.parquet")))
    candidates.extend(_densegen_usr_record_candidates(runbook))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return tuple(deduped)


def _candidate_attempt_paths_for_resume(workspace_root: Path) -> tuple[Path, ...]:
    tables_root = workspace_root / "outputs" / "tables"
    candidate_dirs = [tables_root]
    nested_tables_root = tables_root / "tables"
    if nested_tables_root.exists():
        candidate_dirs.append(nested_tables_root)
    candidates: list[Path] = []
    for directory in candidate_dirs:
        candidates.append(directory / "attempts.parquet")
        candidates.extend(sorted(directory.glob("attempts_part-*.parquet")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return tuple(deduped)


def _orphan_artifact_paths(
    workspace_root: Path,
    *,
    markers: Sequence[str],
) -> tuple[Path, ...]:
    candidate_paths = tuple((workspace_root / marker).resolve() for marker in markers)
    return tuple(path for path in candidate_paths if path.exists())


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise ValueError("resume mode blocked: pyarrow is required for resume readiness checks") from exc
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise ValueError(f"resume mode blocked: unable to read parquet file: {path}") from exc
    metadata = parquet_file.metadata
    if metadata is None:
        return 0
    return int(metadata.num_rows)


def _missing_required_resume_columns(path: Path, *, required_columns: Sequence[str]) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise ValueError("resume mode blocked: pyarrow is required for resume schema checks") from exc
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise ValueError(f"resume mode blocked: unable to read parquet file: {path}") from exc
    available_columns = set(parquet_file.schema_arrow.names)
    missing = tuple(column for column in required_columns if column not in available_columns)
    return missing


def _classify_resume_state(
    runbook: OrchestrationRunbookV1,
    *,
    policy: ResumeReadinessPolicy,
) -> tuple[ResumeState, str]:
    workspace_root = runbook.workspace_root
    run_manifest = workspace_root / "outputs" / "meta" / "run_manifest.json"
    if run_manifest.exists():
        return "resume_ready", f"resume-ready via run manifest: {run_manifest}"

    zero_row_attempt_paths: list[Path] = []
    for path in _candidate_attempt_paths_for_resume(workspace_root):
        if not path.exists():
            continue
        try:
            row_count = _parquet_row_count(path)
        except ValueError as exc:
            return "partial", str(exc)
        if row_count > 0:
            return "resume_ready", f"resume-ready via non-empty attempts artifact: {path} rows={row_count}"
        zero_row_attempt_paths.append(path)

    zero_row_paths: list[Path] = []
    for path in _candidate_record_paths_for_resume(runbook):
        if not path.exists():
            continue
        try:
            row_count = _parquet_row_count(path)
        except ValueError as exc:
            return "partial", str(exc)
        if row_count > 0:
            try:
                missing_columns = _missing_required_resume_columns(
                    path,
                    required_columns=policy.required_record_columns,
                )
            except ValueError as exc:
                return "partial", str(exc)
            if missing_columns:
                missing_summary = ", ".join(missing_columns)
                return (
                    "partial",
                    f"resume records missing required {policy.tool} columns at {path}: {missing_summary}",
                )
            return "resume_ready", f"resume-ready via non-empty records: {path} rows={row_count}"
        zero_row_paths.append(path)

    if zero_row_attempt_paths:
        joined = ", ".join(str(path) for path in zero_row_attempt_paths)
        return "partial", f"zero-row attempts artifacts found: {joined}"
    if zero_row_paths:
        joined = ", ".join(str(path) for path in zero_row_paths)
        return "partial", f"zero-row records found: {joined}"
    orphan_paths = _orphan_artifact_paths(workspace_root, markers=policy.orphan_artifact_markers)
    if orphan_paths:
        joined = ", ".join(str(path) for path in orphan_paths)
        return "partial", f"orphan {policy.tool} artifacts found: {joined}"
    return "none", "missing run manifest and non-empty records"


def resolve_mode_decision(
    *,
    runbook: OrchestrationRunbookV1,
    requested_mode: RunMode | None,
    active_job_ids: Sequence[str],
    runtime_visibility: RuntimeVisibility | None = None,
    allow_fresh_reset: bool = False,
    allow_unknown_active_jobs: bool = False,
) -> ModeDecision:
    selected_requested_mode = requested_mode or runbook.mode_policy.default
    tool_adapter = resolve_mode_tool_adapter(runbook)
    workflow_tool = tool_adapter.tool
    resume_policy = resolve_resume_readiness_policy(workflow_tool)
    has_explicit_resume_policy = resume_policy is not None
    try:
        artifacts_found = tool_adapter.has_resume_artifacts(runbook)
    except InferModeProbeError as exc:
        if selected_requested_mode == "auto":
            raise ValueError(
                "auto mode blocked: infer resume destination is ambiguous or incomplete "
                f"({exc}). Choose --mode explicitly before re-running."
            ) from exc
        if selected_requested_mode == "resume":
            raise ValueError(
                f"resume mode blocked: infer resume destination is ambiguous or incomplete ({exc})."
            ) from exc
        artifacts_found = False
    resume_state: ResumeState = "none"
    resume_readiness_reason = "not-evaluated"
    if has_explicit_resume_policy:
        assert resume_policy is not None
        resume_state, resume_readiness_reason = _classify_resume_state(
            runbook,
            policy=resume_policy,
        )
        artifacts_found = resume_state != "none"
    resume_ready = resume_state == "resume_ready"

    if selected_requested_mode == "auto":
        if has_explicit_resume_policy:
            if resume_state == "none":
                selected_mode = "fresh"
            elif resume_state == "resume_ready":
                selected_mode = "resume"
            else:
                raise ValueError(
                    "auto mode blocked: resume artifacts exist but workspace is not resume-ready "
                    f"({resume_readiness_reason}). "
                    "Choose --mode fresh explicitly only after reviewing workspace state."
                )
        elif not artifacts_found:
            selected_mode = "fresh"
        else:
            selected_mode = "resume"
    else:
        selected_mode = selected_requested_mode

    if selected_mode == "resume" and has_explicit_resume_policy and not resume_ready:
        raise ValueError(f"resume mode blocked: workspace is not resume-ready ({resume_readiness_reason}).")
    if selected_mode == "fresh" and has_explicit_resume_policy and artifacts_found and not allow_fresh_reset:
        raise ValueError(
            "fresh mode blocked: workspace already has resume artifacts "
            f"({resume_readiness_reason}). "
            "Re-run with --allow-fresh-reset only after confirming outputs should be cleared."
        )
    if selected_mode == "resume" and not artifacts_found:
        raise ValueError("resume mode blocked: workspace has no resume artifacts.")
    if selected_mode == "fresh" and artifacts_found and not allow_fresh_reset:
        raise ValueError(
            "fresh mode blocked: workspace already has resume artifacts. "
            "Re-run with --allow-fresh-reset only after confirming outputs should be cleared."
        )

    run_args = tool_adapter.run_args_for_mode(runbook, selected_mode)

    hold_jid: str | None = None
    submit_behavior: SubmitBehavior = "submit"
    reason = f"selected_mode={selected_mode}"
    if has_explicit_resume_policy:
        reason = f"{reason}; resume_ready={str(resume_ready).lower()}"
        if selected_mode == "fresh":
            reason = f"{reason}; fresh_reset_ack={str(allow_fresh_reset).lower()}"

    selected_runtime_visibility = runtime_visibility or _default_runtime_visibility_for_job_ids(active_job_ids)
    hold_jid_candidates = _normalize_hold_jid(active_job_ids)
    if hold_jid_candidates is not None:
        if runbook.mode_policy.on_active_job == "hold_jid":
            submit_behavior = "hold_jid"
            hold_jid = hold_jid_candidates
            reason = f"{reason}; active_jobs_detected; submission_chained_with_hold_jid={hold_jid}"
        else:
            submit_behavior = "blocked"
            reason = f"{reason}; active_jobs_detected; submission_blocked_by_policy"
    elif selected_runtime_visibility.active_job_resolution_state == ActiveJobResolutionState.UNKNOWN:
        if selected_runtime_visibility.scheduler_probe_state == SchedulerProbeState.HOST_DENIED:
            submit_behavior = "blocked"
            reason = f"{reason}; current_host_not_submit_host"
        elif allow_unknown_active_jobs:
            reason = f"{reason}; active_job_visibility_unknown; submission_override_allow_unknown_active_jobs=true"
        else:
            submit_behavior = "blocked"
            reason = f"{reason}; active_job_visibility_unknown; submission_blocked_by_runtime_visibility"
    else:
        reason = f"{reason}; active_job_visibility={selected_runtime_visibility.active_job_resolution_state.value}"

    return ModeDecision(
        requested_mode=selected_requested_mode,
        selected_mode=selected_mode,
        run_args=run_args,
        resume_artifacts_found=artifacts_found,
        submit_behavior=submit_behavior,
        hold_jid=hold_jid,
        reason=reason,
    )
