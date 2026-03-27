"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/checks/runbook.py

Generic ops-runbook-plan preflight check executors.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.preflight.check_protocols import RunbookPlanCheckTarget

from ..models import CommandExecution, PreflightCheck, build_command_check


@dataclass(frozen=True)
class RunbookPlanCheckDependencies:
    run_preflight_command: Callable[..., CommandExecution]
    safe_json_loads: Callable[[str | None], dict[str, object] | None]
    choose_command_summary: Callable[..., str]


def build_runbook_plan_checks(
    *,
    repo_root: Path,
    targets: Sequence[RunbookPlanCheckTarget],
    dependencies: RunbookPlanCheckDependencies,
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    for target in targets:
        execution = dependencies.run_preflight_command(
            (
                "uv",
                "run",
                "ops",
                "runbook",
                "plan",
                "--runbook",
                str(target.runbook_path),
                "--repo-root",
                str(repo_root),
            ),
            cwd=repo_root,
        )
        payload = dependencies.safe_json_loads(execution.stdout) if execution.returncode == 0 else None
        details: dict[str, object] = {
            "runbook": str(target.runbook_path),
            **dict(target.details or {}),
        }
        if isinstance(payload, dict):
            details.update(
                {
                    "selected_mode": payload.get("selected_mode"),
                    "workflow_id": payload.get("workflow_id"),
                    "notify_secret_ref": dict(payload.get("orchestration_notify") or {}).get("secret_ref"),
                }
            )
        checks.append(
            build_command_check(
                check_id=target.check_id,
                kind="runbook_plan",
                required=target.required,
                check_group=target.check_group,
                phase=target.phase,
                phase_id=target.phase_id,
                summary=dependencies.choose_command_summary(
                    execution,
                    fallback=target.fallback_summary,
                ),
                execution=execution,
                surface_id=target.surface_id,
                details=details,
            )
        )
    return tuple(checks)


__all__ = [
    "RunbookPlanCheckDependencies",
    "build_runbook_plan_checks",
]
