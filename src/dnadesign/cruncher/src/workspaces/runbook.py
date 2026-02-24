"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/workspaces/runbook.py

Strict schema/load/execute contracts for machine workspace runbooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml
from pydantic import field_validator, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel

_STEP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ALLOWED_COMMANDS = {
    "analyze",
    "cache",
    "catalog",
    "config",
    "discover",
    "doctor",
    "export",
    "fetch",
    "lock",
    "optimizers",
    "parse",
    "portfolio",
    "runs",
    "sample",
    "sources",
    "status",
    "study",
    "targets",
    "workspaces",
}


class WorkspaceRunbookStep(StrictBaseModel):
    id: str
    run: list[str]
    cwd: Path | None = None
    description: str | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        token = str(value).strip()
        if not token:
            raise ValueError("runbook.steps[].id must be non-empty")
        if not _STEP_ID_RE.match(token):
            raise ValueError("runbook.steps[].id must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
        return token

    @field_validator("run")
    @classmethod
    def _check_run(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("runbook.steps[].run must be non-empty")
        cleaned: list[str] = []
        for idx, item in enumerate(value):
            token = str(item).strip()
            if not token:
                raise ValueError("runbook.steps[].run entries must be non-empty strings")
            if "\n" in token or "\x00" in token:
                raise ValueError("runbook.steps[].run entries must not include control characters")
            if idx == 0 and token not in _ALLOWED_COMMANDS:
                raise ValueError(
                    "runbook.steps[].run uses a disallowed cruncher command. "
                    f"Allowed roots: {sorted(_ALLOWED_COMMANDS)}"
                )
            cleaned.append(token)

        if len(cleaned) >= 2 and cleaned[0] == "workspaces" and cleaned[1] == "run":
            raise ValueError("runbook steps cannot invoke 'workspaces run' recursively")

        return cleaned

    @field_validator("cwd")
    @classmethod
    def _check_cwd(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("runbook.steps[].cwd must be non-empty when provided")
        path = Path(text)
        if path.is_absolute():
            raise ValueError("runbook.steps[].cwd must be relative to the workspace root")
        if ".." in path.parts:
            raise ValueError("runbook.steps[].cwd must stay inside the workspace root")
        return path


class WorkspaceRunbookV1(StrictBaseModel):
    schema_version: int = 1
    name: str
    steps: list[WorkspaceRunbookStep]

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: int) -> int:
        if int(value) != 1:
            raise ValueError("workspace runbook schema v1 required (runbook.schema_version: 1)")
        return int(value)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("runbook.name must be non-empty")
        if not _STEP_ID_RE.match(text):
            raise ValueError("runbook.name must be slug-safe ([A-Za-z0-9][A-Za-z0-9._-]*)")
        return text

    @model_validator(mode="after")
    def _check_steps(self) -> "WorkspaceRunbookV1":
        if not self.steps:
            raise ValueError("runbook.steps must be non-empty")
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError("runbook.steps ids must be unique")
            seen.add(step.id)
        return self


class WorkspaceRunbookRoot(StrictBaseModel):
    runbook: WorkspaceRunbookV1


@dataclass(frozen=True)
class WorkspaceRunbookExecutionResult:
    runbook_path: Path
    workspace_root: Path
    executed_step_ids: list[str]


def _workspace_root_from_runbook_path(runbook_path: Path) -> Path:
    if runbook_path.parent.name == "configs":
        return runbook_path.parent.parent.resolve()
    return runbook_path.parent.resolve()


def _resolve_step_cwd(*, workspace_root: Path, step: WorkspaceRunbookStep) -> Path:
    if step.cwd is None:
        return workspace_root
    candidate = (workspace_root / step.cwd).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"runbook step cwd escapes workspace root: step={step.id!r} cwd={candidate}") from exc
    if not candidate.exists():
        raise FileNotFoundError(f"runbook step cwd does not exist: step={step.id!r} cwd={candidate}")
    if not candidate.is_dir():
        raise ValueError(f"runbook step cwd must be a directory: step={step.id!r} cwd={candidate}")
    return candidate


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        tempfile.TemporaryFile(dir=path).close()
        return True
    except Exception:
        return False


def _runbook_subprocess_env(*, workspace_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    home_raw = str(env.get("HOME", "")).strip()
    home_path = Path(home_raw).expanduser() if home_raw else Path.home()
    if not _is_writable_directory(home_path):
        runtime_home = (workspace_root / ".cruncher" / ".runtime_home").resolve()
        runtime_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(runtime_home)
    return env


def _select_steps(runbook: WorkspaceRunbookV1, step_ids: Sequence[str] | None) -> list[WorkspaceRunbookStep]:
    if not step_ids:
        return list(runbook.steps)
    requested = [str(item).strip() for item in step_ids]
    if any(not item for item in requested):
        raise ValueError("runbook step ids must be non-empty")
    requested_set = set(requested)
    available = {step.id for step in runbook.steps}
    missing = sorted(requested_set - available)
    if missing:
        raise ValueError(f"runbook requested step ids not found: {missing}")
    return [step for step in runbook.steps if step.id in requested_set]


def load_workspace_runbook(path: Path, *, raw: dict | None = None) -> WorkspaceRunbookV1:
    if raw is None:
        if not path.exists():
            raise FileNotFoundError(f"Workspace runbook not found: {path}")
        raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "runbook" not in raw:
        raise ValueError("workspace runbook schema v1 required (missing root key: runbook)")
    payload = raw.get("runbook")
    if not isinstance(payload, dict):
        raise ValueError("workspace runbook schema v1 required (runbook must be a mapping)")
    try:
        return WorkspaceRunbookRoot.model_validate(raw).runbook
    except Exception as exc:
        raise ValueError(str(exc)) from exc


def run_workspace_runbook(
    runbook_path: Path,
    *,
    step_ids: Sequence[str] | None = None,
    dry_run: bool = False,
    output_log_path: Path | None = None,
) -> WorkspaceRunbookExecutionResult:
    resolved_runbook = runbook_path.expanduser().resolve()
    runbook = load_workspace_runbook(resolved_runbook)
    workspace_root = _workspace_root_from_runbook_path(resolved_runbook)
    steps = _select_steps(runbook, step_ids)
    if not steps:
        raise ValueError("runbook resolved zero steps to execute")
    resolved_output_log: Path | None = None
    if output_log_path is not None:
        resolved_output_log = output_log_path.expanduser().resolve()
        resolved_output_log.parent.mkdir(parents=True, exist_ok=True)
    runbook_env = _runbook_subprocess_env(workspace_root=workspace_root)

    executed_step_ids: list[str] = []
    for step in steps:
        step_cwd = _resolve_step_cwd(workspace_root=workspace_root, step=step)
        cmd = ["uv", "run", "cruncher", *step.run]
        if not dry_run:
            if resolved_output_log is None:
                result = subprocess.run(
                    cmd,
                    cwd=str(step_cwd),
                    env=runbook_env,
                    check=False,
                )
            else:
                with resolved_output_log.open("a", encoding="utf-8") as log_handle:
                    log_handle.write(f"\n=== step: {step.id} ===\n")
                    log_handle.write(f"cwd: {step_cwd}\n")
                    log_handle.write(f"command: {' '.join(cmd)}\n")
                    log_handle.flush()
                    result = subprocess.run(
                        cmd,
                        cwd=str(step_cwd),
                        env=runbook_env,
                        check=False,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
            if int(result.returncode) != 0:
                if resolved_output_log is not None:
                    raise RuntimeError(
                        "Runbook step failed: "
                        f"step={step.id!r} returncode={result.returncode} command={' '.join(cmd)} "
                        f"(see step log: {resolved_output_log})"
                    )
                raise RuntimeError(
                    "Runbook step failed: "
                    f"step={step.id!r} returncode={result.returncode} command={' '.join(cmd)} "
                    "(see streamed step output above)"
                )
        executed_step_ids.append(step.id)

    return WorkspaceRunbookExecutionResult(
        runbook_path=resolved_runbook,
        workspace_root=workspace_root,
        executed_step_ids=executed_step_ids,
    )
