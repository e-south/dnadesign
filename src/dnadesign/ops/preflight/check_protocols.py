"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/check_protocols.py

Shared check-target contracts for generic OPS preflight executors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EnvironmentMatchMode = Literal["all", "any"]


@dataclass(frozen=True)
class CommandCheckTarget:
    check_id: str
    check_group: str | None
    category: str
    check_set_id: str | None
    argv: tuple[str, ...]
    cwd: Path
    fallback_summary: str
    required: bool = True
    surface_id: str | None = None
    details: Mapping[str, object] | None = None


@dataclass(frozen=True)
class RunbookPlanCheckTarget:
    check_id: str
    check_group: str
    category: str
    check_set_id: str | None
    runbook_path: Path
    fallback_summary: str
    required: bool = True
    surface_id: str | None = None
    details: Mapping[str, object] | None = None


@dataclass(frozen=True)
class EnvironmentCheckTarget:
    check_id: str
    check_group: str | None
    category: str
    check_set_id: str | None
    flag_names: tuple[str, ...]
    match_mode: EnvironmentMatchMode
    ok_summary: str
    missing_summary: str
    required: bool = True
    details: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SchedulerQueueCheckTarget:
    check_id: str
    check_group: str | None
    category: str
    check_set_id: str | None
    backend: str
    max_running_jobs: int
    max_queued_jobs: int | None = None
    required: bool = True
    surface_id: str | None = None
    details: Mapping[str, object] | None = None


__all__ = [
    "CommandCheckTarget",
    "EnvironmentCheckTarget",
    "EnvironmentMatchMode",
    "RunbookPlanCheckTarget",
    "SchedulerQueueCheckTarget",
]
