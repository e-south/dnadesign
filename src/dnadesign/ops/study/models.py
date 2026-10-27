"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/study/models.py

Generic study status service contracts for OPS.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

StudySummaryScope = Literal["repo", "workspace", "host", "cluster"]
StudyPreflightScope = Literal["next", "full"]

STUDY_SUMMARY_SCOPES = frozenset({"repo", "workspace", "host", "cluster"})
STUDY_PREFLIGHT_SCOPES = frozenset({"next", "full"})


@dataclass(frozen=True)
class StudyPreflightContract:
    default_scope: StudyPreflightScope
    scope_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    scope_payloads: dict[str, dict[str, object]] = field(default_factory=dict)
    check_specs: dict[str, tuple[dict[str, object], ...]] = field(default_factory=dict)

    @property
    def known_groups(self) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for group in (group for groups in self.scope_groups.values() for group in groups):
            normalized_group = str(group).strip()
            if normalized_group and normalized_group not in seen:
                seen.add(normalized_group)
                ordered.append(normalized_group)
        return tuple(ordered)


@dataclass(frozen=True)
class StudyOpsContract:
    study_id: str
    status_kind: str | None
    preflight_kind: str | None
    snapshot_summary_scope: StudySummaryScope
    preflight: StudyPreflightContract
    title: str | None = None
    record_sources: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, dict[str, object]] = field(default_factory=dict)
    execution_surfaces: dict[str, dict[str, object]] = field(default_factory=dict)
    raw_payload: dict[str, object] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class StudyStatusContext:
    repo_root: Path
    study_root: Path
    contract: StudyOpsContract
    service_context: object


class StudyStatusService(Protocol):
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
    "STUDY_PREFLIGHT_SCOPES",
    "STUDY_SUMMARY_SCOPES",
    "StudyOpsContract",
    "StudyPreflightContract",
    "StudyPreflightScope",
    "StudySummaryScope",
    "StudyStatusService",
    "StudyStatusContext",
]
