"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/check_protocols.py

Shared check-target contracts for generic OPS preflight executors.

Module Author(s): Codex
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
    phase: str
    phase_id: str | None
    argv: tuple[str, ...]
    cwd: Path
    fallback_summary: str
    details: Mapping[str, object] | None = None


@dataclass(frozen=True)
class RunbookPlanCheckTarget:
    check_id: str
    check_group: str
    phase: str
    phase_id: str | None
    runbook_path: Path
    fallback_summary: str
    details: Mapping[str, object] | None = None


@dataclass(frozen=True)
class EnvironmentCheckTarget:
    check_id: str
    check_group: str | None
    phase: str
    phase_id: str | None
    flag_names: tuple[str, ...]
    match_mode: EnvironmentMatchMode
    ok_summary: str
    missing_summary: str
    details: Mapping[str, object] | None = None


__all__ = [
    "CommandCheckTarget",
    "EnvironmentCheckTarget",
    "EnvironmentMatchMode",
    "RunbookPlanCheckTarget",
]
