"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/models.py

Generic study status adapter contracts for OPS.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

StudyPhaseStatus = Literal["planned", "ready", "in_progress", "complete", "blocked", "blocked_gpu", "parallel_optional"]
StudyLifecycleMode = Literal["sequential", "tracks"]
StudySummaryScope = Literal["repo", "workspace", "host", "cluster"]
StudyPreflightScope = Literal["next", "full"]

STUDY_PHASE_STATUSES = frozenset(
    {"planned", "ready", "in_progress", "complete", "blocked", "blocked_gpu", "parallel_optional"}
)
STUDY_LIFECYCLE_MODES = frozenset({"sequential", "tracks"})
STUDY_SUMMARY_SCOPES = frozenset({"repo", "workspace", "host", "cluster"})
STUDY_PREFLIGHT_SCOPES = frozenset({"next", "full"})


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
    default_scope: StudyPreflightScope
    group_phase_bindings: dict[str, str] = field(default_factory=dict)
    next_scope: StudyPreflightNextScopeContract = field(default_factory=StudyPreflightNextScopeContract)
    scope_payloads: dict[str, dict[str, object]] = field(default_factory=dict)
    check_specs: dict[str, tuple[dict[str, object], ...]] = field(default_factory=dict)

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
class StudyPhaseContract:
    id: str
    status: StudyPhaseStatus
    next_surface: str | None = None
    blocker: str | None = None
    output_dataset: str | None = None
    primary_dataset: str | None = None
    required_for_main_study_state: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "next_surface": self.next_surface,
            "blocker": self.blocker,
            "output_dataset": self.output_dataset,
            "primary_dataset": self.primary_dataset,
            "required_for_main_study_state": self.required_for_main_study_state,
        }


@dataclass(frozen=True)
class StudyOpsContract:
    study_id: str
    status_kind: str
    preflight_kind: str
    phase_order: tuple[str, ...]
    snapshot_summary_scope: StudySummaryScope
    preflight: StudyPreflightContract
    lifecycle_mode: StudyLifecycleMode = "sequential"
    lifecycle_item_label: str = "phase"
    title: str | None = None
    record_sources: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, dict[str, object]] = field(default_factory=dict)
    execution_surfaces: dict[str, dict[str, object]] = field(default_factory=dict)
    current_phase_id: str | None = None
    phases: tuple[StudyPhaseContract, ...] = ()
    raw_payload: dict[str, object] = field(default_factory=dict, repr=False)

    @property
    def phase_states(self) -> tuple[dict[str, object], ...]:
        return tuple(phase.as_dict() for phase in self.phases)

    @property
    def phase_index(self) -> dict[str, StudyPhaseContract]:
        return {phase.id: phase for phase in self.phases}


@dataclass(frozen=True)
class StudyStatusContext:
    repo_root: Path
    study_root: Path
    contract: StudyOpsContract
    adapter_context: object


class StudyStatusAdapter(Protocol):
    status_kind: str

    def load_context(self, *, repo_root: Path | None, study_root: Path | None) -> StudyStatusContext: ...

    def build_snapshot(self, context: StudyStatusContext) -> tuple[str, str, dict[str, object]]: ...

    def build_preflight(
        self,
        context: StudyStatusContext,
        *,
        scope: str | None,
    ) -> tuple[str, str, dict[str, object]]: ...


__all__ = [
    "STUDY_LIFECYCLE_MODES",
    "STUDY_PHASE_STATUSES",
    "STUDY_PREFLIGHT_SCOPES",
    "STUDY_SUMMARY_SCOPES",
    "StudyLifecycleMode",
    "StudyOpsContract",
    "StudyPhaseContract",
    "StudyPhaseStatus",
    "StudyPreflightContract",
    "StudyPreflightNextScopeContract",
    "StudyPreflightScope",
    "StudySummaryScope",
    "StudyStatusAdapter",
    "StudyStatusContext",
]
