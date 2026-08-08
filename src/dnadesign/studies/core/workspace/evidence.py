"""Load and verify study evidence indexes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .contracts import StudyArtifact, StudyEvidenceIndex
from .validation import (
    artifact_uri,
    identifier,
    mapping,
    optional_text,
    reject_unknown_keys,
    relative_file,
    require_keys,
    sequence,
    sha256_digest,
    string_mapping,
    text,
)

_INDEX_KEYS = frozenset({"schema", "study_id", "artifacts"})
_ARTIFACT_KEYS = frozenset(
    {
        "artifact_id",
        "artifact_type",
        "status",
        "path",
        "uri",
        "media_type",
        "content_digest",
        "source_revisions",
        "generated_by",
        "blocker",
    }
)
_ARTIFACT_REQUIRED_KEYS = frozenset({"artifact_id", "artifact_type", "status", "source_revisions"})
_MATERIALIZED_STATUSES = frozenset({"available", "stale", "superseded"})


def load_study_evidence_index(
    index_path: Path,
    *,
    study_root: Path,
    expected_study_id: str,
) -> StudyEvidenceIndex:
    """Load one evidence index and verify every tracked artifact digest."""

    resolved_study_root = study_root.expanduser().resolve()
    resolved_index_path = index_path.expanduser().resolve()
    _require_contained(resolved_index_path, resolved_study_root, label="evidence index")
    try:
        payload = yaml.safe_load(resolved_index_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"could not read evidence index {resolved_index_path}: {exc}") from exc
    index = mapping(payload, label="evidence index")
    reject_unknown_keys(index, allowed=_INDEX_KEYS, label="evidence index")
    require_keys(index, required=_INDEX_KEYS, label="evidence index")
    schema = text(index.get("schema"), label="evidence index schema")
    if schema != "study-evidence-index/v1":
        raise ValueError(f"unsupported evidence index schema {schema!r}: {resolved_index_path}")
    study_id = identifier(index.get("study_id"), label="evidence index study_id")
    if study_id != expected_study_id:
        raise ValueError(f"evidence index study_id {study_id!r} does not match expected study_id {expected_study_id!r}")

    artifacts: list[StudyArtifact] = []
    seen_ids: set[str] = set()
    for position, raw_artifact in enumerate(
        sequence(index.get("artifacts"), label="evidence index artifacts", allow_empty=True),
        start=1,
    ):
        artifact = _load_artifact(
            raw_artifact,
            position=position,
            evidence_root=resolved_index_path.parent,
            study_root=resolved_study_root,
        )
        if artifact.artifact_id in seen_ids:
            raise ValueError(f"evidence index has duplicate artifact_id {artifact.artifact_id!r}")
        seen_ids.add(artifact.artifact_id)
        artifacts.append(artifact)

    return StudyEvidenceIndex(
        schema=schema,
        study_id=study_id,
        path=resolved_index_path,
        artifacts=tuple(artifacts),
    )


def _load_artifact(
    value: object,
    *,
    position: int,
    evidence_root: Path,
    study_root: Path,
) -> StudyArtifact:
    label = f"evidence artifact {position}"
    payload = mapping(value, label=label)
    reject_unknown_keys(payload, allowed=_ARTIFACT_KEYS, label=label)
    require_keys(payload, required=_ARTIFACT_REQUIRED_KEYS, label=label)
    artifact_id = identifier(payload.get("artifact_id"), label=f"{label}.artifact_id")
    artifact_type = identifier(payload.get("artifact_type"), label=f"{label}.artifact_type")
    status = text(payload.get("status"), label=f"{label}.status")
    if status not in {*_MATERIALIZED_STATUSES, "blocked"}:
        raise ValueError(f"{label} has unsupported status {status!r}")
    source_revisions = string_mapping(payload.get("source_revisions"), label=f"{label}.source_revisions")
    media_type = optional_text(payload.get("media_type"), label=f"{label}.media_type")
    blocker = optional_text(payload.get("blocker"), label=f"{label}.blocker")

    if status == "blocked":
        forbidden = {"path", "uri", "content_digest", "generated_by", "media_type"}.intersection(payload)
        if blocker is None or forbidden:
            raise ValueError(f"{label}: blocked artifact requires blocker and must not define a location or output")
        return StudyArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            status="blocked",
            source_revisions=source_revisions,
            blocker=blocker,
        )

    has_path = "path" in payload
    has_uri = "uri" in payload
    if has_path == has_uri:
        raise ValueError(f"{label} must define exactly one of path or uri")
    require_keys(payload, required=frozenset({"content_digest", "generated_by"}), label=label)
    content_digest = sha256_digest(payload.get("content_digest"), label=f"{label}.content_digest")
    generated_by = _command(payload.get("generated_by"), label=f"{label}.generated_by")
    path = None
    uri = None
    if has_path:
        path = relative_file(
            base=evidence_root,
            value=payload.get("path"),
            boundary=study_root,
            label=f"{label}.path",
        )
        observed = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != content_digest:
            raise ValueError(
                f"{label} content digest mismatch for {path}: expected {content_digest}, observed {observed}"
            )
    else:
        uri = artifact_uri(payload.get("uri"), label=f"{label}.uri")

    return StudyArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        status=status,  # type: ignore[arg-type]
        source_revisions=source_revisions,
        path=path,
        uri=uri,
        media_type=media_type,
        content_digest=content_digest,
        generated_by=generated_by,
        blocker=blocker,
    )


def _command(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return tuple(text(item, label=f"{label}[{index}]") for index, item in enumerate(value))


def _require_contained(path: Path, boundary: Path, *, label: str) -> None:
    try:
        path.relative_to(boundary)
    except ValueError as exc:
        raise ValueError(f"{label} escapes study root {boundary}: {path}") from exc


__all__ = ["load_study_evidence_index"]
