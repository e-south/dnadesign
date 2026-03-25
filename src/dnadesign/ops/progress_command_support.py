"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/progress_command_support.py

Command execution and preflight helper support for ops progress providers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .progress_support import (
    load_yaml_mapping,
    required_metadata_text,
    resolve_input_path,
    string_or_none,
)


@dataclass(frozen=True)
class CommandExecution:
    argv: tuple[str, ...]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def render_argv(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in argv)


def build_infer_notify_setup_command(*, config_path: Path) -> str:
    return render_argv(
        (
            "uv",
            "run",
            "notify",
            "setup",
            "slack",
            "--tool",
            "infer",
            "--config",
            str(config_path),
            "--secret-source",
            "file",
            "--secret-ref",
            "file://$NOTIFY_WEBHOOK_FILE",
        )
    )


def run_progress_command(argv: Sequence[str], *, cwd: Path, timeout_seconds: int = 180) -> CommandExecution:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandExecution(
            argv=tuple(str(token) for token in argv),
            cwd=str(cwd),
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or ""),
            timed_out=True,
        )
    return CommandExecution(
        argv=tuple(str(token) for token in argv),
        cwd=str(cwd),
        returncode=int(completed.returncode),
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
        timed_out=False,
    )


def preflight_command_check(
    *,
    check_id: str,
    check_group: str | None,
    phase: str,
    phase_id: str | None,
    summary: str,
    execution: CommandExecution,
    details: dict[str, object] | None = None,
    override_state: str | None = None,
) -> dict[str, object]:
    state = override_state
    if state is None:
        state = "attention" if execution.returncode != 0 or execution.timed_out else "ok"
    if execution.timed_out:
        summary = f"timed out: {summary}"
    return {
        "id": check_id,
        "check_group": str(check_group or "").strip() or None,
        "phase": phase,
        "phase_id": phase_id,
        "state": state,
        "summary": summary,
        "command": render_argv(execution.argv),
        "cwd": execution.cwd,
        "returncode": execution.returncode,
        "timed_out": execution.timed_out,
        "stdout_tail": _trim_command_output(execution.stdout),
        "stderr_tail": _trim_command_output(execution.stderr),
        "details": details or {},
    }


def preflight_state_check(
    *,
    check_id: str,
    check_group: str | None,
    phase: str,
    phase_id: str | None,
    state: str,
    summary: str,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": check_id,
        "check_group": str(check_group or "").strip() or None,
        "phase": phase,
        "phase_id": phase_id,
        "state": state,
        "summary": summary,
        "command": None,
        "cwd": None,
        "returncode": None,
        "timed_out": False,
        "stdout_tail": None,
        "stderr_tail": None,
        "details": details or {},
    }


def choose_command_summary(execution: CommandExecution, *, fallback: str) -> str:
    candidate_texts = (
        (execution.stderr, execution.stdout) if execution.returncode != 0 else (execution.stdout, execution.stderr)
    )
    for text in candidate_texts:
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in {"{", "}", "[", "]"}:
                continue
            if stripped.startswith("WARNING:") or stripped.startswith("W0000 "):
                continue
            return stripped
    return fallback


def safe_json_loads(text: str | None) -> dict[str, object] | None:
    payload = str(text or "").strip()
    if not payload:
        return None
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def load_orchestration_runbook_payload(path: Path) -> dict[str, object]:
    payload = load_yaml_mapping(path, label="runbook yaml")
    runbook = payload.get("runbook") or {}
    if not isinstance(runbook, dict):
        raise ValueError(f"runbook yaml must define a 'runbook' mapping: {path}")
    return runbook


def infer_usr_dataset_requirements(config_path: Path) -> list[dict[str, object]]:
    payload = load_yaml_mapping(config_path, label="infer config")
    jobs_payload = payload.get("jobs") or []
    if jobs_payload and not isinstance(jobs_payload, list):
        raise ValueError(f"infer config must define a 'jobs' list: {config_path}")
    requirements: list[dict[str, object]] = []
    for job_payload in jobs_payload:
        if not isinstance(job_payload, dict):
            raise ValueError(f"infer config job entry must be a mapping: {config_path}")
        ingest_payload = job_payload.get("ingest") or {}
        if not isinstance(ingest_payload, dict):
            raise ValueError(f"infer config job ingest must be a mapping: {config_path}")
        if string_or_none(ingest_payload.get("source")) != "usr":
            continue
        dataset_id = required_metadata_text(ingest_payload.get("dataset"), label="ingest dataset", source=config_path)
        raw_root = required_metadata_text(ingest_payload.get("root"), label="ingest root", source=config_path)
        root_path = resolve_input_path(Path(raw_root), base_dir=config_path.parent)
        records_path = (root_path / dataset_id / "records.parquet").resolve()
        requirements.append(
            {
                "job_id": string_or_none(job_payload.get("id")),
                "dataset": dataset_id,
                "usr_root": str(root_path),
                "records_path": str(records_path),
                "exists": records_path.exists(),
            }
        )
    return requirements


def _trim_command_output(text: str | None, *, max_lines: int = 8, max_chars: int = 1200) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    lines = raw.splitlines()
    if len(lines) > max_lines:
        raw = "\n".join(lines[-max_lines:])
    if len(raw) > max_chars:
        raw = raw[-max_chars:]
    return raw


__all__ = [
    "CommandExecution",
    "build_infer_notify_setup_command",
    "choose_command_summary",
    "infer_usr_dataset_requirements",
    "load_orchestration_runbook_payload",
    "preflight_command_check",
    "preflight_state_check",
    "render_argv",
    "run_progress_command",
    "safe_json_loads",
]
