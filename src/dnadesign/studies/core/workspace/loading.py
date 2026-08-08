"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/workspace/loading.py

Load a portable study workspace from one explicit repository root.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .contracts import StudyCatalogProgram, StudyManifest, StudyWorkflow, StudyWorkspace
from .evidence import load_study_evidence_index
from .validation import (
    identifier,
    iso_date,
    mapping,
    reject_unknown_keys,
    relative_file,
    require_keys,
    sequence,
    text,
)

_CATALOG_KEYS = frozenset({"schema", "programs", "studies"})
_PROGRAM_KEYS = frozenset({"program_id", "title", "entrypoint"})
_CATALOG_STUDY_KEYS = frozenset({"study_id", "manifest"})
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "study_id",
        "program_id",
        "title",
        "summary",
        "visibility",
        "status",
        "owners",
        "last_verified",
        "entrypoint",
        "operations",
        "evidence_index",
        "workflows",
    }
)
_MANIFEST_REQUIRED_KEYS = _MANIFEST_KEYS.difference({"operations"})
_WORKFLOW_KEYS = frozenset({"tool_id", "route", "requires"})
_VISIBILITIES = frozenset({"private", "public", "restricted"})
_STATUSES = frozenset({"planned", "active", "paused", "complete", "archived"})


def load_study_workspace(root: Path) -> StudyWorkspace:
    """Load a strict ``study-catalog/v1`` rooted at ``catalog/studies.yaml``."""

    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"study workspace root does not exist: {resolved_root}")
    catalog_path = resolved_root / "catalog" / "studies.yaml"
    try:
        payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"could not read study catalog {catalog_path}: {exc}") from exc
    catalog = mapping(payload, label="study catalog")
    reject_unknown_keys(catalog, allowed=_CATALOG_KEYS, label="study catalog")
    require_keys(catalog, required=_CATALOG_KEYS, label="study catalog")
    schema = text(catalog.get("schema"), label="study catalog schema")
    if schema != "study-catalog/v1":
        raise ValueError(f"unsupported study catalog schema {schema!r}: {catalog_path}")

    programs = _load_programs(catalog.get("programs"), root=resolved_root)
    program_ids = {program.program_id for program in programs}
    studies = _load_studies(
        catalog.get("studies"),
        root=resolved_root,
        program_ids=program_ids,
    )
    return StudyWorkspace(
        schema=schema,
        root=resolved_root,
        catalog_path=catalog_path,
        programs=programs,
        studies=studies,
    )


def _load_programs(value: object, *, root: Path) -> tuple[StudyCatalogProgram, ...]:
    programs: list[StudyCatalogProgram] = []
    seen: set[str] = set()
    for position, raw_program in enumerate(sequence(value, label="study catalog programs"), start=1):
        label = f"study catalog program {position}"
        payload = mapping(raw_program, label=label)
        reject_unknown_keys(payload, allowed=_PROGRAM_KEYS, label=label)
        require_keys(payload, required=_PROGRAM_KEYS, label=label)
        program_id = identifier(payload.get("program_id"), label=f"{label}.program_id")
        if program_id in seen:
            raise ValueError(f"study catalog has duplicate program_id {program_id!r}")
        seen.add(program_id)
        programs.append(
            StudyCatalogProgram(
                program_id=program_id,
                title=text(payload.get("title"), label=f"{label}.title"),
                entrypoint=relative_file(
                    base=root,
                    value=payload.get("entrypoint"),
                    boundary=root,
                    label=f"{label}.entrypoint",
                ),
            )
        )
    return tuple(programs)


def _load_studies(
    value: object,
    *,
    root: Path,
    program_ids: set[str],
) -> tuple[StudyManifest, ...]:
    studies: list[StudyManifest] = []
    seen: set[str] = set()
    for position, raw_study in enumerate(sequence(value, label="study catalog studies"), start=1):
        label = f"study catalog entry {position}"
        payload = mapping(raw_study, label=label)
        reject_unknown_keys(payload, allowed=_CATALOG_STUDY_KEYS, label=label)
        require_keys(payload, required=_CATALOG_STUDY_KEYS, label=label)
        study_id = identifier(payload.get("study_id"), label=f"{label}.study_id")
        if study_id in seen:
            raise ValueError(f"study catalog has duplicate study_id {study_id!r}")
        seen.add(study_id)
        manifest_path = relative_file(
            base=root,
            value=payload.get("manifest"),
            boundary=root,
            label=f"{label}.manifest",
        )
        studies.append(
            _load_manifest(
                manifest_path,
                root=root,
                expected_study_id=study_id,
                program_ids=program_ids,
            )
        )
    return tuple(studies)


def _load_manifest(
    manifest_path: Path,
    *,
    root: Path,
    expected_study_id: str,
    program_ids: set[str],
) -> StudyManifest:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest = mapping(payload, label=f"study manifest {manifest_path}")
    reject_unknown_keys(manifest, allowed=_MANIFEST_KEYS, label="study manifest")
    require_keys(manifest, required=_MANIFEST_REQUIRED_KEYS, label="study manifest")
    schema = text(manifest.get("schema"), label="study manifest schema")
    if schema != "study/v1":
        raise ValueError(f"unsupported study manifest schema {schema!r}: {manifest_path}")
    study_id = identifier(manifest.get("study_id"), label="study manifest study_id")
    if study_id != expected_study_id:
        raise ValueError(f"manifest study_id {study_id!r} does not match catalog study_id {expected_study_id!r}")
    program_id = identifier(manifest.get("program_id"), label="study manifest program_id")
    if program_id not in program_ids:
        raise ValueError(f"study {study_id!r} references undeclared program_id {program_id!r}")
    visibility = text(manifest.get("visibility"), label="study manifest visibility")
    if visibility not in _VISIBILITIES:
        raise ValueError(f"study manifest has unsupported visibility {visibility!r}")
    status = text(manifest.get("status"), label="study manifest status")
    if status not in _STATUSES:
        raise ValueError(f"study manifest has unsupported status {status!r}")
    owners = _unique_identifiers(manifest.get("owners"), label="study manifest owners")
    study_root = manifest_path.parent
    entrypoint = relative_file(
        base=study_root,
        value=manifest.get("entrypoint"),
        boundary=study_root,
        label=f"study {study_id} entrypoint",
    )
    evidence_path = relative_file(
        base=study_root,
        value=manifest.get("evidence_index"),
        boundary=study_root,
        label=f"study {study_id} evidence_index",
    )
    operations = None
    if "operations" in manifest:
        operations = relative_file(
            base=study_root,
            value=manifest.get("operations"),
            boundary=study_root,
            label=f"study {study_id} operations",
        )
    workflows = _load_workflows(manifest.get("workflows"), study_root=study_root, study_id=study_id)
    return StudyManifest(
        schema=schema,
        study_id=study_id,
        program_id=program_id,
        title=text(manifest.get("title"), label="study manifest title"),
        summary=text(manifest.get("summary"), label="study manifest summary"),
        visibility=visibility,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        owners=owners,
        last_verified=iso_date(manifest.get("last_verified"), label="study manifest last_verified"),
        root=study_root,
        manifest_path=manifest_path,
        entrypoint=entrypoint,
        operations=operations,
        evidence=load_study_evidence_index(
            evidence_path,
            study_root=study_root,
            expected_study_id=study_id,
        ),
        workflows=workflows,
    )


def _load_workflows(value: object, *, study_root: Path, study_id: str) -> tuple[StudyWorkflow, ...]:
    workflows: list[StudyWorkflow] = []
    seen: set[str] = set()
    for position, raw_workflow in enumerate(
        sequence(value, label=f"study {study_id} workflows", allow_empty=True),
        start=1,
    ):
        label = f"study {study_id} workflow {position}"
        payload = mapping(raw_workflow, label=label)
        reject_unknown_keys(payload, allowed=_WORKFLOW_KEYS, label=label)
        require_keys(payload, required=_WORKFLOW_KEYS, label=label)
        tool_id = identifier(payload.get("tool_id"), label=f"{label}.tool_id")
        if tool_id in seen:
            raise ValueError(f"study {study_id} has duplicate workflow tool_id {tool_id!r}")
        seen.add(tool_id)
        try:
            route = relative_file(
                base=study_root,
                value=payload.get("route"),
                boundary=study_root,
                label=f"workflow {tool_id} route",
            )
        except ValueError as exc:
            if "does not exist" in str(exc):
                raise ValueError(f"workflow {tool_id} route does not exist") from exc
            raise
        workflows.append(
            StudyWorkflow(
                tool_id=tool_id,
                route=route,
                requires=text(payload.get("requires"), label=f"{label}.requires"),
            )
        )
    return tuple(workflows)


def _unique_identifiers(value: object, *, label: str) -> tuple[str, ...]:
    identifiers = tuple(
        identifier(item, label=f"{label}[{index}]") for index, item in enumerate(sequence(value, label=label))
    )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} must not contain duplicates")
    return identifiers


__all__ = ["load_study_workspace"]
