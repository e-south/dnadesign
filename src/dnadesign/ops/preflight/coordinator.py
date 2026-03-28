"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/preflight/coordinator.py

Generic preflight-check scope filtering and blocker evaluation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from dnadesign.ops.status.models import STATE_SEVERITY, state_counts

from .models import PreflightCheck


class PreflightScopePlan(Protocol):
    scope: str
    target_phase_id: str | None
    included_groups: Sequence[str]
    phase_scoped_groups: Sequence[str]


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
    phase_states: Sequence[Mapping[str, object]] | None,
    scope_plan: PreflightScopePlan,
) -> PreflightCheckEvaluation:
    scoped_checks = tuple(check for check in checks if check_matches_scope(check, scope_plan=scope_plan))
    phase_status_index = _phase_status_index(phase_states)
    scoped_counts = _counts_by_state(scoped_checks)
    all_counts = _counts_by_state(checks)
    blocker_checks = _ordered_preflight_blockers(
        scoped_checks,
        phase_status_index=phase_status_index,
        scope_plan=scope_plan,
    )
    deferred_blockers = _ordered_preflight_blockers(
        [check for check in checks if check not in scoped_checks],
        phase_status_index=phase_status_index,
        scope_plan=scope_plan,
    )
    nonblocking_attention_checks = _ordered_nonblocking_preflight_attention(
        scoped_checks,
        phase_status_index=phase_status_index,
        scope_plan=scope_plan,
    )
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
    if scope_plan.scope != "next" or scope_plan.target_phase_id is None:
        return True
    group = str(check.check_group or "").strip()
    if not group:
        raise ValueError(f"study preflight check is missing check_group: {check.id!r}")
    if group not in scope_plan.included_groups:
        return False
    if group not in scope_plan.phase_scoped_groups:
        return True
    return str(check.phase_id or "").strip() == scope_plan.target_phase_id


def _counts_by_state(checks: Sequence[PreflightCheck]) -> dict[str, int]:
    return state_counts(check.state for check in checks)


def _ordered_preflight_blockers(
    checks: Sequence[PreflightCheck],
    *,
    phase_status_index: Mapping[str, str],
    scope_plan: PreflightScopePlan,
) -> tuple[PreflightCheck, ...]:
    failing = [check for check in checks if check.state != "ok"]
    failing = [
        check
        for check in failing
        if _preflight_check_is_blocking(
            check,
            phase_status_index=phase_status_index,
            scope_plan=scope_plan,
        )
    ]
    return tuple(
        sorted(
            failing,
            key=lambda check: _preflight_order_key(check, scope_plan=scope_plan),
        )
    )


def _ordered_nonblocking_preflight_attention(
    checks: Sequence[PreflightCheck],
    *,
    phase_status_index: Mapping[str, str],
    scope_plan: PreflightScopePlan,
) -> tuple[PreflightCheck, ...]:
    nonblocking = [
        check
        for check in checks
        if check.state != "ok"
        and not _preflight_check_is_blocking(
            check,
            phase_status_index=phase_status_index,
            scope_plan=scope_plan,
        )
    ]
    return tuple(
        sorted(
            nonblocking,
            key=lambda check: _preflight_order_key(check, scope_plan=scope_plan),
        )
    )


def _preflight_order_key(
    check: PreflightCheck,
    *,
    scope_plan: PreflightScopePlan,
) -> tuple[int, int, int, str]:
    severity_rank = -STATE_SEVERITY.get(check.state, -1)
    if scope_plan.scope != "next" or scope_plan.target_phase_id is None:
        return (severity_rank, 0, 0, check.id)

    check_group = str(check.check_group or "").strip()
    group_is_shared = check_group not in scope_plan.phase_scoped_groups
    shared_rank = 0 if group_is_shared else 1
    phase_rank = 0 if str(check.phase_id or "").strip() == scope_plan.target_phase_id else 1
    return (severity_rank, shared_rank, phase_rank, check.id)


def _preflight_check_is_blocking(
    check: PreflightCheck,
    *,
    phase_status_index: Mapping[str, str],
    scope_plan: PreflightScopePlan,
) -> bool:
    if not check.required:
        return False
    phase_id = str(check.phase_id or "").strip()
    if not phase_id:
        return True
    if scope_plan.scope == "next" and scope_plan.target_phase_id is not None:
        check_group = str(check.check_group or "").strip()
        if check_group and check_group not in scope_plan.phase_scoped_groups:
            return True
    phase_status = phase_status_index.get(phase_id)
    return phase_status not in {"complete", "parallel_optional"}


def _phase_status_index(phase_states: Sequence[Mapping[str, object]] | None) -> dict[str, str]:
    if phase_states is None:
        return {}
    result: dict[str, str] = {}
    for phase in phase_states:
        phase_id = str(phase.get("id") or "").strip()
        phase_status = str(phase.get("status") or "").strip()
        if phase_id and phase_status:
            result[phase_id] = phase_status
    return result


__all__ = [
    "PreflightCheckEvaluation",
    "check_matches_scope",
    "evaluate_preflight_checks",
]
