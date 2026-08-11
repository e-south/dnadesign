"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/coordinator.py

Generic preflight-check scope filtering and blocker evaluation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dnadesign.ops.status.models import STATE_SEVERITY, state_counts

from .models import PreflightCheck


class PreflightScopePlan(Protocol):
    scope: str
    included_groups: Sequence[str]


@dataclass(frozen=True)
class PreflightCheckEvaluation:
    scoped_checks: tuple[PreflightCheck, ...]
    blocker_checks: tuple[PreflightCheck, ...]
    deferred_blockers: tuple[PreflightCheck, ...]
    nonblocking_attention_checks: tuple[PreflightCheck, ...]
    scoped_counts: dict[str, int]
    effective_counts: dict[str, int]

    @property
    def blocked_by_ids(self) -> tuple[str, ...]:
        return tuple(check.id for check in self.blocker_checks)

    @property
    def deferred_check_ids(self) -> tuple[str, ...]:
        return tuple(check.id for check in self.deferred_blockers)

    @property
    def nonblocking_attention_ids(self) -> tuple[str, ...]:
        return tuple(check.id for check in self.nonblocking_attention_checks)


def evaluate_preflight_checks(
    checks: Sequence[PreflightCheck],
    *,
    scope_plan: PreflightScopePlan,
) -> PreflightCheckEvaluation:
    scoped_checks = tuple(check for check in checks if check_matches_scope(check, scope_plan=scope_plan))
    scoped_counts = _counts_by_state(scoped_checks)
    all_counts = _counts_by_state(checks)
    blocker_checks = _ordered_preflight_blockers(scoped_checks)
    deferred_blockers = _ordered_preflight_blockers(
        [check for check in checks if check not in scoped_checks],
    )
    nonblocking_attention_checks = _ordered_nonblocking_preflight_attention(scoped_checks)
    effective_counts = scoped_counts if scope_plan.scope == "next" else all_counts
    return PreflightCheckEvaluation(
        scoped_checks=scoped_checks,
        blocker_checks=blocker_checks,
        deferred_blockers=deferred_blockers,
        nonblocking_attention_checks=nonblocking_attention_checks,
        scoped_counts=scoped_counts,
        effective_counts=effective_counts,
    )


def check_matches_scope(
    check: PreflightCheck,
    *,
    scope_plan: PreflightScopePlan,
) -> bool:
    if scope_plan.scope != "next":
        return True
    group = str(check.check_group or "").strip()
    if not group:
        raise ValueError(f"study preflight check is missing check_group: {check.id!r}")
    if group not in scope_plan.included_groups:
        return False
    return True


def _counts_by_state(checks: Sequence[PreflightCheck]) -> dict[str, int]:
    return state_counts(check.state for check in checks)


def _ordered_preflight_blockers(
    checks: Sequence[PreflightCheck],
) -> tuple[PreflightCheck, ...]:
    failing = [check for check in checks if check.state != "ok"]
    failing = [check for check in failing if check.required]
    return tuple(
        sorted(
            failing,
            key=_preflight_order_key,
        )
    )


def _ordered_nonblocking_preflight_attention(
    checks: Sequence[PreflightCheck],
) -> tuple[PreflightCheck, ...]:
    nonblocking = [check for check in checks if check.state != "ok" and not check.required]
    return tuple(
        sorted(
            nonblocking,
            key=_preflight_order_key,
        )
    )


def _preflight_order_key(
    check: PreflightCheck,
) -> tuple[int, str]:
    severity_rank = -STATE_SEVERITY.get(check.state, -1)
    return (severity_rank, check.id)


__all__ = [
    "PreflightCheckEvaluation",
    "check_matches_scope",
    "evaluate_preflight_checks",
]
