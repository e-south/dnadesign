"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/state.py

Deterministic run-mode and active-job submission-behavior resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from dnadesign._contracts import (
    ResumeReadinessPolicy,
    resolve_resume_readiness_policy,
    resolve_usr_producer_contract,
)

from ..runbooks.schema import OrchestrationRunbookV1

RunMode = Literal["auto", "fresh", "resume"]
SubmitBehavior = Literal["submit", "hold_jid", "blocked"]
ResumeState = Literal["none", "resume_ready", "partial"]


def _run_probe(argv: Sequence[str]) -> tuple[int, str, str]:
    result = subprocess.run(list(argv), check=False, capture_output=True, text=True)
    return int(result.returncode), result.stdout, result.stderr


def _parse_job_ids_from_qstat_output(text: str) -> tuple[str, ...]:
    job_ids: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        job_ids.append(parts[0])
    return tuple(job_ids)


def discover_active_job_ids_for_runbook(
    runbook: OrchestrationRunbookV1,
    *,
    max_jobs: int = 24,
) -> tuple[str, ...]:
    if max_jobs <= 0:
        return ()

    user = os.environ.get("USER", "")
    return_code, stdout, stderr = _run_probe(("qstat", "-u", user))
    if return_code != 0:
        message = stderr.strip() or stdout.strip() or "qstat -u failed"
        raise RuntimeError(message)

    tokens: list[str] = [str(runbook.workspace_root)]
    if runbook.densegen is not None:
        tokens.append(str(runbook.densegen.config))
    if runbook.infer is not None:
        tokens.append(str(runbook.infer.config))
    if runbook.notify is not None:
        tokens.append(str(runbook.notify.profile))
    unique_tokens = tuple(dict.fromkeys(token for token in tokens if token))

    active_job_ids: list[str] = []
    for job_id in _parse_job_ids_from_qstat_output(stdout)[:max_jobs]:
        rc, job_stdout, _job_stderr = _run_probe(("qstat", "-j", str(job_id)))
        if rc != 0:
            continue
        if any(token in job_stdout for token in unique_tokens):
            active_job_ids.append(str(job_id))
    return tuple(active_job_ids)


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


def _infer_overlay_artifacts(workspace_root: Path, *, infer_config: Path | None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    usr_root = workspace_root / "outputs" / "usr_datasets"
    if usr_root.exists():
        candidates.extend(sorted(usr_root.glob("**/_derived/infer.parquet")))
        candidates.extend(sorted(usr_root.glob("**/_derived/infer/*.parquet")))

    if infer_config is not None:
        contract = _resolve_infer_usr_output_for_mode_probe(infer_config)
        if contract is not None:
            dataset_root = contract.usr_root / contract.usr_dataset
            candidates.append(dataset_root / "_derived" / "infer.parquet")
            infer_parts_root = dataset_root / "_derived" / "infer"
            if infer_parts_root.exists():
                candidates.extend(sorted(infer_parts_root.glob("*.parquet")))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return tuple(deduped)


def _resolve_infer_usr_output_for_mode_probe(infer_config: Path):
    try:
        return resolve_usr_producer_contract(tool="infer", config_path=infer_config)
    except ValueError as exc:
        message = str(exc)
        if "at least one job with ingest.source='usr' and io.write_back=true" in message:
            return None
        raise ValueError(
            "infer mode probe requires a single resolvable USR destination in infer config "
            f"{infer_config}: {message}"
        ) from exc


def _has_resume_artifacts(runbook: OrchestrationRunbookV1, *, workflow_tool: str) -> bool:
    workspace_root = runbook.workspace_root
    tool = str(workflow_tool or "").strip().lower()
    if tool == "infer":
        manifest_path = workspace_root / "outputs" / "meta" / "run_manifest.json"
        if manifest_path.exists():
            return True
        infer_config = runbook.infer.config if runbook.infer is not None else None
        return bool(_infer_overlay_artifacts(workspace_root, infer_config=infer_config))

    markers = (
        workspace_root / "outputs" / "meta" / "run_manifest.json",
        workspace_root / "outputs" / "tables" / "records.parquet",
        workspace_root / "outputs" / "usr_datasets" / "registry.yaml",
    )
    if any(path.exists() for path in markers):
        return True
    tables_root = workspace_root / "outputs" / "tables"
    candidate_dirs = [tables_root]
    nested_tables_root = tables_root / "tables"
    if nested_tables_root.exists():
        candidate_dirs.append(nested_tables_root)
    for directory in candidate_dirs:
        if any(directory.glob("records__part-*.parquet")):
            return True
        if any(directory.glob("attempts_part-*.parquet")):
            return True
    return False


def _candidate_record_paths_for_resume(workspace_root: Path) -> tuple[Path, ...]:
    tables_root = workspace_root / "outputs" / "tables"
    candidate_dirs = [tables_root]
    nested_tables_root = tables_root / "tables"
    if nested_tables_root.exists():
        candidate_dirs.append(nested_tables_root)
    candidates: list[Path] = []
    for directory in candidate_dirs:
        candidates.append(directory / "records.parquet")
        candidates.extend(sorted(directory.glob("records__part-*.parquet")))
    usr_root = workspace_root / "outputs" / "usr_datasets"
    if usr_root.exists():
        candidates.extend(sorted(usr_root.glob("**/records.parquet")))
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
    workspace_root: Path,
    *,
    policy: ResumeReadinessPolicy,
) -> tuple[ResumeState, str]:
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
    for path in _candidate_record_paths_for_resume(workspace_root):
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
    allow_fresh_reset: bool = False,
) -> ModeDecision:
    selected_requested_mode = requested_mode or runbook.mode_policy.default
    workflow_tool = "densegen" if runbook.densegen is not None else "infer"
    resume_policy = resolve_resume_readiness_policy(workflow_tool)
    has_explicit_resume_policy = resume_policy is not None
    artifacts_found = _has_resume_artifacts(runbook, workflow_tool=workflow_tool)
    resume_state: ResumeState = "none"
    resume_readiness_reason = "not-evaluated"
    if has_explicit_resume_policy:
        assert resume_policy is not None
        resume_state, resume_readiness_reason = _classify_resume_state(
            runbook.workspace_root,
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

    if runbook.densegen is not None:
        assert runbook.densegen is not None
        if selected_mode == "fresh":
            run_args = runbook.densegen.run_args.fresh
        else:
            run_args = runbook.densegen.run_args.resume
    else:
        run_args = ""

    hold_jid: str | None = None
    submit_behavior: SubmitBehavior = "submit"
    reason = f"selected_mode={selected_mode}"
    if has_explicit_resume_policy:
        reason = f"{reason}; resume_ready={str(resume_ready).lower()}"
        if selected_mode == "fresh":
            reason = f"{reason}; fresh_reset_ack={str(allow_fresh_reset).lower()}"

    hold_jid_candidates = _normalize_hold_jid(active_job_ids)
    if hold_jid_candidates is not None:
        if runbook.mode_policy.on_active_job == "hold_jid":
            submit_behavior = "hold_jid"
            hold_jid = hold_jid_candidates
            reason = f"{reason}; active_jobs_detected; submission_chained_with_hold_jid={hold_jid}"
        else:
            submit_behavior = "blocked"
            reason = f"{reason}; active_jobs_detected; submission_blocked_by_policy"

    return ModeDecision(
        requested_mode=selected_requested_mode,
        selected_mode=selected_mode,
        run_args=run_args,
        resume_artifacts_found=artifacts_found,
        submit_behavior=submit_behavior,
        hold_jid=hold_jid,
        reason=reason,
    )
