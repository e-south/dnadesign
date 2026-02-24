"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/discovery.py

Discover workspace-scoped Study specs and deterministic Study run directories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.study.layout import STUDY_OUTPUT_ROOT, study_manifest_path, study_status_path
from dnadesign.cruncher.study.load import load_study_spec
from dnadesign.cruncher.study.manifest import load_study_manifest, load_study_status

_STUDY_SPECS_RELATIVE_DIR = Path("configs/studies")


@dataclass(frozen=True)
class StudySpecRecord:
    workspace_name: str
    workspace_root: Path
    spec_path: Path
    study_name: str


@dataclass(frozen=True)
class StudyRunRecord:
    workspace_name: str
    workspace_root: Path
    study_name: str
    study_id: str
    status: str
    run_dir: Path


def discover_study_specs_for_workspace(*, workspace_name: str, workspace_root: Path) -> list[StudySpecRecord]:
    rows: list[StudySpecRecord] = []
    seen_names: dict[str, Path] = {}
    specs_root = (workspace_root / _STUDY_SPECS_RELATIVE_DIR).resolve()
    if not specs_root.is_dir():
        return rows
    for spec_path in sorted(specs_root.glob("*.study.yaml")):
        spec = load_study_spec(spec_path)
        prior = seen_names.get(spec.name)
        if prior is not None:
            raise ValueError(
                f"Duplicate study.name '{spec.name}' in workspace {workspace_root}: {prior} and {spec_path.resolve()}"
            )
        seen_names[spec.name] = spec_path.resolve()
        rows.append(
            StudySpecRecord(
                workspace_name=workspace_name,
                workspace_root=workspace_root.resolve(),
                spec_path=spec_path.resolve(),
                study_name=spec.name,
            )
        )
    return rows


def discover_study_runs_for_workspace(specs: list[StudySpecRecord]) -> list[StudyRunRecord]:
    rows: list[StudyRunRecord] = []
    for spec in specs:
        study_root = (spec.workspace_root / STUDY_OUTPUT_ROOT / spec.study_name).resolve()
        if not study_root.is_dir():
            continue
        for candidate in sorted(study_root.iterdir()):
            if not candidate.is_dir():
                continue
            manifest_file = study_manifest_path(candidate)
            status_file = study_status_path(candidate)
            has_manifest = manifest_file.exists()
            has_status = status_file.exists()
            if not has_manifest and not has_status:
                continue
            if not has_manifest or not has_status:
                raise ValueError(
                    f"Study run directory is missing required metadata files: {candidate} "
                    "(requires both study_manifest.json and study_status.json)."
                )
            manifest = load_study_manifest(manifest_file)
            status = load_study_status(status_file)
            if manifest.study_name != spec.study_name:
                raise ValueError(
                    f"Study run name mismatch for {candidate}: "
                    f"manifest.study_name={manifest.study_name!r}, expected={spec.study_name!r}."
                )
            if manifest.study_id != candidate.name:
                raise ValueError(
                    f"Study run id mismatch for {candidate}: "
                    f"manifest.study_id={manifest.study_id!r}, expected directory name {candidate.name!r}."
                )
            rows.append(
                StudyRunRecord(
                    workspace_name=spec.workspace_name,
                    workspace_root=spec.workspace_root,
                    study_name=manifest.study_name,
                    study_id=manifest.study_id,
                    status=status.status,
                    run_dir=candidate.resolve(),
                )
            )
    rows.sort(key=lambda row: (row.workspace_name, row.study_name, row.study_id))
    return rows
