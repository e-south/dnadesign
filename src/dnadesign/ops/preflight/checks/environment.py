"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/checks/environment.py

Generic environment-flag preflight check executors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dnadesign.ops.preflight.check_protocols import EnvironmentCheckTarget

from ..models import PreflightCheck, build_state_check


def build_environment_checks(
    *,
    flag_state: Mapping[str, bool],
    targets: Sequence[EnvironmentCheckTarget],
) -> tuple[PreflightCheck, ...]:
    checks: list[PreflightCheck] = []
    resolved_flag_state = {name: bool(value) for name, value in flag_state.items()}
    for target in targets:
        required_flags = tuple(str(name).strip() for name in target.flag_names if str(name).strip())
        if not required_flags:
            raise ValueError(f"environment check {target.check_id!r} must declare at least one flag name")
        if target.match_mode == "all":
            matched = all(resolved_flag_state.get(flag_name, False) for flag_name in required_flags)
        elif target.match_mode == "any":
            matched = any(resolved_flag_state.get(flag_name, False) for flag_name in required_flags)
        else:
            raise ValueError(f"environment check {target.check_id!r} has unsupported match_mode {target.match_mode!r}")
        checks.append(
            build_state_check(
                check_id=target.check_id,
                kind="environment",
                required=target.required,
                check_group=target.check_group,
                category=target.category,
                check_set_id=target.check_set_id,
                state="ok" if matched else "attention",
                summary=target.ok_summary if matched else target.missing_summary,
                details={
                    **resolved_flag_state,
                    **dict(target.details or {}),
                },
            )
        )
    return tuple(checks)


__all__ = ["build_environment_checks"]
