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
class StudyOpsContract:
    study_id: str
    family: str
    phase_order: tuple[str, ...]
    snapshot_summary_scope: str
    preflight_default_scope: str
    preflight_phase_targets: dict[str, str]
    next_scope_phase_groups: dict[str, tuple[str, ...]]
    infer_lane_groups: tuple[str, ...] = field(default_factory=tuple)
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


__all__ = ["StudyFamilyAdapter", "StudyOpsContract", "StudyStatusContext"]
