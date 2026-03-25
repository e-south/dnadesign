"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/preflight_plan.py

Generic study-preflight scope planning and blocker evaluation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .models import StudyPreflightContract

_STATE_PRIORITY = {"missing": 0, "attention": 1}


@dataclass(frozen=True)
class StudyPreflightPlan:
    scope: str
    target_phase_id: str | None
    included_groups: tuple[str, ...]
    phase_scoped_groups: tuple[str, ...] = ()

    def includes_group(self, group: str) -> bool:
        return str(group).strip() in self.included_groups


@dataclass(frozen=True)
class StudyPreflightCheckEvaluation:
    scoped_checks: tuple[Mapping[str, object], ...]
    blocker_checks: tuple[Mapping[str, object], ...]
    deferred_blockers: tuple[Mapping[str, object], ...]
    nonblocking_attention_checks: tuple[Mapping[str, object], ...]
    scoped_counts: dict[str, int]
    effective_counts: dict[str, int]

    @property
    def blocked_by_ids(self) -> tuple[str, ...]:
        return tuple(str(check["id"]) for check in self.blocker_checks)

    @property
    def deferred_check_ids(self) -> tuple[str, ...]:
        return tuple(str(check["id"]) for check in self.deferred_blockers)

    @property
    def nonblocking_attention_ids(self) -> tuple[str, ...]:
        return tuple(str(check["id"]) for check in self.nonblocking_attention_checks)


def normalize_study_preflight_scope(scope: str | None, *, default_scope: str = "full") -> str:
    normalized = str(scope or default_scope).strip().lower()
    if normalized not in {"full", "next"}:
        raise ValueError("study preflight scope must be one of: full, next")
    return normalized


def build_study_preflight_plan(
    *,
    current_phase: str | None,
    next_ready_phase: Mapping[str, object] | None,
    scope: str | None,
    contract: StudyPreflightContract,
    runtime_phase_ids: Sequence[str] | None = None,
) -> StudyPreflightPlan:
    scope_norm = normalize_study_preflight_scope(scope, default_scope=contract.default_scope)
    target_phase_id = _target_phase_id(
        current_phase=current_phase,
        next_ready_phase=next_ready_phase,
        scope=scope_norm,
    )
    if scope_norm != "next":
        return StudyPreflightPlan(
            scope=scope_norm,
            target_phase_id=target_phase_id,
            included_groups=contract.known_groups,
        )

    if target_phase_id is None:
        raise ValueError("study preflight next scope requires a current or next-ready phase")

    runtime_phase_index = {str(phase_id).strip() for phase_id in runtime_phase_ids or () if str(phase_id).strip()}
    if target_phase_id in contract.next_scope.target_phase_groups:
        included_groups = contract.next_scope.target_phase_groups[target_phase_id]
        return StudyPreflightPlan(
            scope=scope_norm,
            target_phase_id=target_phase_id,
            included_groups=included_groups,
        )

    if target_phase_id in runtime_phase_index:
        included_groups = _ordered_unique(
            (*contract.next_scope.runtime_shared_groups, *contract.next_scope.runtime_phase_groups)
        )
        return StudyPreflightPlan(
            scope=scope_norm,
            target_phase_id=target_phase_id,
            included_groups=included_groups,
            phase_scoped_groups=contract.next_scope.runtime_phase_groups,
        )

    raise ValueError(f"ops.study.yaml does not declare next-scope groups for phase {target_phase_id!r}")


def evaluate_study_preflight_checks(
    checks: Sequence[Mapping[str, object]],
    *,
    phase_states: Sequence[Mapping[str, object]] | None,
    scope_plan: StudyPreflightPlan,
) -> StudyPreflightCheckEvaluation:
    scoped_checks = tuple(
        check for check in checks if study_preflight_check_matches_scope(check, scope_plan=scope_plan)
    )
    scoped_counts = _counts_by_state(scoped_checks)
    all_counts = _counts_by_state(checks)
    blocker_checks = _ordered_preflight_blockers(
        scoped_checks,
        phase_states=phase_states,
        scope=scope_plan.scope,
    )
    deferred_blockers = _ordered_preflight_blockers(
        [check for check in checks if check not in scoped_checks],
        phase_states=phase_states,
        scope=scope_plan.scope,
    )
    nonblocking_attention_checks = _ordered_nonblocking_preflight_attention(
        scoped_checks,
        phase_states=phase_states,
        scope=scope_plan.scope,
    )
    effective_counts = scoped_counts if scope_plan.scope == "next" else all_counts
    return StudyPreflightCheckEvaluation(
        scoped_checks=scoped_checks,
        blocker_checks=blocker_checks,
        deferred_blockers=deferred_blockers,
        nonblocking_attention_checks=nonblocking_attention_checks,
        scoped_counts=scoped_counts,
        effective_counts=effective_counts,
    )


def study_preflight_check_matches_scope(
    check: Mapping[str, object],
    *,
    scope_plan: StudyPreflightPlan,
) -> bool:
    if scope_plan.scope != "next" or scope_plan.target_phase_id is None:
        return True
    group = str(check.get("check_group") or "").strip()
    if not group:
        raise ValueError(f"study preflight check is missing check_group: {check.get('id')!r}")
    if group not in scope_plan.included_groups:
        return False
    if group not in scope_plan.phase_scoped_groups:
        return True
    return str(check.get("phase_id") or "").strip() == scope_plan.target_phase_id


def _counts_by_state(checks: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(check["state"]) for check in checks)
    return {state: int(counts.get(state, 0)) for state in ("ok", "attention", "missing")}


def _target_phase_id(
    *,
    current_phase: str | None,
    next_ready_phase: Mapping[str, object] | None,
    scope: str,
) -> str | None:
    if scope != "next":
        return None
    if next_ready_phase is not None:
        phase_id = str(next_ready_phase.get("id") or "").strip()
        return phase_id or None
    phase_id = str(current_phase or "").strip()
    return phase_id or None


def _ordered_preflight_blockers(
    checks: Sequence[Mapping[str, object]],
    *,
    phase_states: Sequence[Mapping[str, object]] | None,
    scope: str,
) -> tuple[Mapping[str, object], ...]:
    failing = [check for check in checks if check["state"] != "ok"]
    if scope == "full":
        phase_status_index = _phase_status_index(phase_states)
        failing = [
            check for check in failing if _preflight_check_is_blocking(check, phase_status_index=phase_status_index)
        ]
    return tuple(
        sorted(
            failing,
            key=lambda check: (
                _STATE_PRIORITY.get(str(check["state"]), 99),
                str(check["id"]),
            ),
        )
    )


def _ordered_nonblocking_preflight_attention(
    checks: Sequence[Mapping[str, object]],
    *,
    phase_states: Sequence[Mapping[str, object]] | None,
    scope: str,
) -> tuple[Mapping[str, object], ...]:
    if scope != "full":
        return ()
    phase_status_index = _phase_status_index(phase_states)
    nonblocking = [
        check
        for check in checks
        if check["state"] != "ok" and not _preflight_check_is_blocking(check, phase_status_index=phase_status_index)
    ]
    return tuple(
        sorted(
            nonblocking,
            key=lambda check: (
                _STATE_PRIORITY.get(str(check["state"]), 99),
                str(check["id"]),
            ),
        )
    )


def _preflight_check_is_blocking(
    check: Mapping[str, object],
    *,
    phase_status_index: Mapping[str, str],
) -> bool:
    phase_id = str(check.get("phase_id") or "").strip()
    if not phase_id:
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


def _ordered_unique(groups: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        normalized_group = str(group).strip()
        if normalized_group and normalized_group not in seen:
            seen.add(normalized_group)
            ordered.append(normalized_group)
    return tuple(ordered)


__all__ = [
    "StudyPreflightCheckEvaluation",
    "StudyPreflightPlan",
    "build_study_preflight_plan",
    "evaluate_study_preflight_checks",
    "normalize_study_preflight_scope",
    "study_preflight_check_matches_scope",
]
