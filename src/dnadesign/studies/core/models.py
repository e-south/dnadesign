"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/models.py

Generic study-family adapter contracts for OPS.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StudyPreflightNextScopeContract:
    target_phase_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    runtime_phase_groups: tuple[str, ...] = field(default_factory=tuple)
    runtime_shared_groups: tuple[str, ...] = field(default_factory=tuple)

    @property
    def known_groups(self) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for group in (
            *(group for groups in self.target_phase_groups.values() for group in groups),
            *self.runtime_shared_groups,
            *self.runtime_phase_groups,
        ):
            normalized_group = str(group).strip()
            if normalized_group and normalized_group not in seen:
                seen.add(normalized_group)
                ordered.append(normalized_group)
        return tuple(ordered)


@dataclass(frozen=True)
class StudyPreflightContract:
    default_scope: str
    group_phase_bindings: dict[str, str] = field(default_factory=dict)
    next_scope: StudyPreflightNextScopeContract = field(default_factory=StudyPreflightNextScopeContract)

    @property
    def known_groups(self) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for group in (*self.group_phase_bindings, *self.next_scope.known_groups):
            normalized_group = str(group).strip()
            if normalized_group and normalized_group not in seen:
                seen.add(normalized_group)
                ordered.append(normalized_group)
        return tuple(ordered)


@dataclass(frozen=True)
class StudyOpsContract:
    study_id: str
    family: str
    phase_order: tuple[str, ...]
    snapshot_summary_scope: str
    preflight: StudyPreflightContract
    raw_payload: dict[str, object] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class StudyStatusContext:
    repo_root: Path
    study_root: Path
    contract: StudyOpsContract
    family_context: object


class StudyFamilyAdapter(Protocol):
    family_id: str

    def load_context(self, *, repo_root: Path | None, study_root: Path | None) -> StudyStatusContext: ...

    def build_snapshot(self, context: StudyStatusContext) -> tuple[str, str, dict[str, object]]: ...

    def build_preflight(
        self,
        context: StudyStatusContext,
        *,
        scope: str | None,
    ) -> tuple[str, str, dict[str, object]]: ...


__all__ = [
    "StudyFamilyAdapter",
    "StudyOpsContract",
    "StudyPreflightContract",
    "StudyPreflightNextScopeContract",
    "StudyStatusContext",
]
