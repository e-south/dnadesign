"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/support.py

Generic command execution and payload helpers for OPS preflight surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from dnadesign.ops.status.artifacts import load_yaml_mapping
from dnadesign.ops.status.parsing import (
    required_metadata_text,
    string_or_none,
)
from dnadesign.ops.status.paths import (
    resolve_input_path,
)

from .models import CommandExecution, render_argv


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


def run_preflight_command(argv: Sequence[str], *, cwd: Path, timeout_seconds: int = 180) -> CommandExecution:
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


__all__ = [
    "build_infer_notify_setup_command",
    "choose_command_summary",
    "infer_usr_dataset_requirements",
    "load_orchestration_runbook_payload",
    "render_argv",
    "run_preflight_command",
    "safe_json_loads",
]
