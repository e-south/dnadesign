"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/preflight_plan.py

Generic study-preflight scope planning and blocker evaluation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .models import StudyPreflightContract


@dataclass(frozen=True)
class StudyPreflightPlan:
    scope: str
    target_phase_id: str | None
    included_groups: tuple[str, ...]
    phase_scoped_groups: tuple[str, ...] = ()

    def includes_group(self, group: str) -> bool:
        return str(group).strip() in self.included_groups


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
    "StudyPreflightPlan",
    "build_study_preflight_plan",
    "normalize_study_preflight_scope",
]
