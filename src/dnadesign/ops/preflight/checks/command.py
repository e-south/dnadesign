"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/checks/command.py

Generic command-backed preflight check executors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from dnadesign.ops.preflight.check_protocols import CommandCheckTarget

from ..models import CommandExecution, PreflightCheck, build_command_check


@dataclass(frozen=True)
class CommandCheckDependencies:
    run_preflight_command: Callable[..., CommandExecution]
    choose_command_summary: Callable[..., str]


def build_command_checks(
    *,
    targets: Sequence[CommandCheckTarget],
    dependencies: CommandCheckDependencies,
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    for target in targets:
        execution = dependencies.run_preflight_command(target.argv, cwd=target.cwd)
        checks.append(
            build_command_check(
                check_id=target.check_id,
                kind="command",
                required=target.required,
                check_group=target.check_group,
                category=target.category,
                check_set_id=target.check_set_id,
                summary=dependencies.choose_command_summary(
                    execution,
                    fallback=target.fallback_summary,
                ),
                execution=execution,
                surface_id=target.surface_id,
                details=dict(target.details or {}),
            )
        )
    return tuple(checks)


__all__ = [
    "CommandCheckDependencies",
    "build_command_checks",
]
