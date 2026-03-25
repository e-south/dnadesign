"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/models.py

Typed command-execution and preflight-check models for OPS readiness surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

_CHECK_STATES = frozenset({"ok", "attention", "missing"})


@dataclass(frozen=True)
class CommandExecution:
    argv: tuple[str, ...]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class PreflightCheck:
    id: str
    phase: str
    state: Literal["ok", "attention", "missing"]
    summary: str
    check_group: str | None = None
    phase_id: str | None = None
    command: str | None = None
    cwd: str | None = None
    returncode: int | None = None
    timed_out: bool = False
    stdout_tail: str | None = None
    stderr_tail: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_id = str(self.id or "").strip()
        normalized_phase = str(self.phase or "").strip()
        normalized_summary = str(self.summary or "").strip()
        if not normalized_id:
            raise ValueError("preflight check id must be non-empty")
        if not normalized_phase:
            raise ValueError(f"preflight check {normalized_id} phase must be non-empty")
        if self.state not in _CHECK_STATES:
            raise ValueError(f"preflight check {normalized_id} has unsupported state {self.state!r}")
        if not normalized_summary:
            raise ValueError(f"preflight check {normalized_id} summary must be non-empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "check_group": self.check_group,
            "phase": self.phase,
            "phase_id": self.phase_id,
            "state": self.state,
            "summary": self.summary,
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "details": dict(self.details),
        }


def build_command_check(
    *,
    check_id: str,
    check_group: str | None,
    phase: str,
    phase_id: str | None,
    summary: str,
    execution: CommandExecution,
    details: dict[str, object] | None = None,
    override_state: Literal["ok", "attention", "missing"] | None = None,
) -> PreflightCheck:
    state = override_state
    if state is None:
        state = "attention" if execution.returncode != 0 or execution.timed_out else "ok"
    if execution.timed_out:
        summary = f"timed out: {summary}"
    return PreflightCheck(
        id=check_id,
        check_group=str(check_group or "").strip() or None,
        phase=phase,
        phase_id=phase_id,
        state=state,
        summary=summary,
        command=render_argv(execution.argv),
        cwd=execution.cwd,
        returncode=execution.returncode,
        timed_out=execution.timed_out,
        stdout_tail=_trim_command_output(execution.stdout),
        stderr_tail=_trim_command_output(execution.stderr),
        details=details or {},
    )


def build_state_check(
    *,
    check_id: str,
    check_group: str | None,
    phase: str,
    phase_id: str | None,
    state: Literal["ok", "attention", "missing"],
    summary: str,
    details: dict[str, object] | None = None,
) -> PreflightCheck:
    return PreflightCheck(
        id=check_id,
        check_group=str(check_group or "").strip() or None,
        phase=phase,
        phase_id=phase_id,
        state=state,
        summary=summary,
        details=details or {},
    )


def render_argv(argv: tuple[str, ...]) -> str:
    from shlex import quote

    return " ".join(quote(str(token)) for token in argv)


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
    "PreflightCheck",
    "build_command_check",
    "build_state_check",
]
