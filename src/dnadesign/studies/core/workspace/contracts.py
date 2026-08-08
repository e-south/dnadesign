"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/workspace/contracts.py

Immutable values exposed by the study workspace contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

StudyVisibility = Literal["private", "public", "restricted"]
StudyStatus = Literal["planned", "active", "paused", "complete", "archived"]
StudyArtifactStatus = Literal["available", "stale", "superseded", "blocked"]


@dataclass(frozen=True, slots=True)
class StudyCatalogProgram:
    """One declared grouping boundary in a research workspace."""

    program_id: str
    title: str
    entrypoint: Path


@dataclass(frozen=True, slots=True)
class StudyWorkflow:
    """A thin route from a study to a versioned tool surface."""

    tool_id: str
    route: Path
    requires: str


@dataclass(frozen=True, slots=True)
class StudyArtifact:
    """One reviewable or explicitly blocked study deliverable."""

    artifact_id: str
    artifact_type: str
    status: StudyArtifactStatus
    source_revisions: dict[str, str]
    path: Path | None = None
    uri: str | None = None
    media_type: str | None = None
    content_digest: str | None = None
    generated_by: tuple[str, ...] = ()
    blocker: str | None = None


@dataclass(frozen=True, slots=True)
class StudyEvidenceIndex:
    """Typed evidence inventory for one study."""

    schema: str
    study_id: str
    path: Path
    artifacts: tuple[StudyArtifact, ...]


@dataclass(frozen=True, slots=True)
class StudyManifest:
    """Portable identity and navigation contract for one study."""

    schema: str
    study_id: str
    program_id: str
    title: str
    summary: str
    visibility: StudyVisibility
    status: StudyStatus
    owners: tuple[str, ...]
    last_verified: str
    root: Path
    manifest_path: Path
    entrypoint: Path
    evidence: StudyEvidenceIndex
    workflows: tuple[StudyWorkflow, ...]
    operations: Path | None = None


@dataclass(frozen=True, slots=True)
class StudyWorkspace:
    """Validated catalog plus all referenced study manifests."""

    schema: str
    root: Path
    catalog_path: Path
    programs: tuple[StudyCatalogProgram, ...]
    studies: tuple[StudyManifest, ...]

    @property
    def program_index(self) -> dict[str, StudyCatalogProgram]:
        return {program.program_id: program for program in self.programs}

    @property
    def study_index(self) -> dict[str, StudyManifest]:
        return {study.study_id: study for study in self.studies}


__all__ = [
    "StudyArtifact",
    "StudyArtifactStatus",
    "StudyCatalogProgram",
    "StudyEvidenceIndex",
    "StudyManifest",
    "StudyStatus",
    "StudyVisibility",
    "StudyWorkflow",
    "StudyWorkspace",
]
