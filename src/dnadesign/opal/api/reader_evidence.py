"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/reader_evidence.py

Public routing contract for digest-bound Reader evidence manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

READER_EVIDENCE_API_VERSION = "1"
READER_EVIDENCE_MANIFEST_ADAPTER = "opal.reader_evidence_manifest.v1"
_SUMMARY_FIELDS = {
    "rows",
    "distinct_ids",
    "reader_experiments",
    "artifact_count",
    "missing_artifact_rows",
}
_ARTIFACT_FIELDS = {
    "semantic_kind",
    "kind",
    "record_id",
    "scope",
    "path",
    "exists",
    "media_type",
}


class ReaderEvidenceManifestAdapterError(ValueError):
    """Raised when a producer manifest violates OPAL's Reader projection."""


@dataclass(frozen=True)
class ReaderEvidenceManifestProjection:
    """Validated producer-neutral fields used by OPAL Reader evidence views."""

    producer_schema_version: str
    round_label: str
    summary: Mapping[str, int]
    rows: tuple[Mapping[str, Any], ...]


def parse_reader_evidence_manifest_adapter(payload: object) -> ReaderEvidenceManifestProjection:
    """Validate and project one producer-owned Reader evidence manifest.

    The producer retains authority over ``schema_version``. OPAL routes only
    manifests that explicitly declare this public adapter, so notebook code
    does not need to know a study's schema identity. The adapter owns the small
    summary, row, and artifact projection that OPAL displays.
    """

    if not isinstance(payload, Mapping):
        raise ReaderEvidenceManifestAdapterError("Reader evidence manifest must be a mapping.")
    if payload.get("opal_adapter") != READER_EVIDENCE_MANIFEST_ADAPTER:
        raise ReaderEvidenceManifestAdapterError(
            f"Reader evidence manifest must declare adapter {READER_EVIDENCE_MANIFEST_ADAPTER!r}."
        )
    producer_schema_version = _text(payload.get("schema_version"), field="schema_version")
    round_label = _text(payload.get("round"), field="round")
    summary = _summary(payload.get("summary"))
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ReaderEvidenceManifestAdapterError("Reader evidence manifest rows must be a list.")

    rows: list[Mapping[str, Any]] = []
    identities: set[str] = set()
    experiments: set[str] = set()
    artifact_count = 0
    missing_artifact_rows = 0
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise ReaderEvidenceManifestAdapterError(f"Reader evidence row {index} must be a mapping.")
        row = dict(raw_row)
        identity = _text(row.get("candidate_id") or row.get("id"), field=f"rows[{index}].id")
        _text(row.get("design_id"), field=f"rows[{index}].design_id")
        experiment = _text(
            row.get("reader_experiment_id"),
            field=f"rows[{index}].reader_experiment_id",
        )
        artifacts = _artifacts(row.get("artifacts"), row_index=index)
        identities.add(identity)
        experiments.add(experiment)
        artifact_count += len(artifacts)
        missing_kinds = row.get("missing_artifact_kinds")
        if missing_kinds is not None and (
            not isinstance(missing_kinds, list)
            or any(not isinstance(value, str) or not value.strip() for value in missing_kinds)
        ):
            raise ReaderEvidenceManifestAdapterError(
                f"Reader evidence row {index} missing_artifact_kinds must be a list of non-empty strings."
            )
        if not artifacts or bool(missing_kinds) or any(not bool(item["exists"]) for item in artifacts):
            missing_artifact_rows += 1
        rows.append(row)

    expected = {
        "rows": len(rows),
        "distinct_ids": len(identities),
        "reader_experiments": len(experiments),
        "artifact_count": artifact_count,
        "missing_artifact_rows": missing_artifact_rows,
    }
    drift = {field: (summary[field], value) for field, value in expected.items() if summary[field] != value}
    if drift:
        raise ReaderEvidenceManifestAdapterError(f"Reader evidence manifest summary disagrees with its rows: {drift}.")
    return ReaderEvidenceManifestProjection(
        producer_schema_version=producer_schema_version,
        round_label=round_label,
        summary=summary,
        rows=tuple(rows),
    )


def _summary(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _SUMMARY_FIELDS:
        raise ReaderEvidenceManifestAdapterError(
            f"Reader evidence manifest summary fields must be exactly {sorted(_SUMMARY_FIELDS)}."
        )
    return {field: _count(value[field], field=f"summary.{field}") for field in sorted(_SUMMARY_FIELDS)}


def _artifacts(value: object, *, row_index: int) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ReaderEvidenceManifestAdapterError(f"Reader evidence row {row_index} artifacts must be a list.")
    artifacts: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for artifact_index, raw_artifact in enumerate(value):
        context = f"rows[{row_index}].artifacts[{artifact_index}]"
        if not isinstance(raw_artifact, Mapping):
            raise ReaderEvidenceManifestAdapterError(f"Reader evidence {context} must be a mapping.")
        missing = sorted(_ARTIFACT_FIELDS - set(raw_artifact))
        if missing:
            raise ReaderEvidenceManifestAdapterError(f"Reader evidence {context} is missing fields: {missing}.")
        semantic_kind = _text(raw_artifact.get("semantic_kind"), field=f"{context}.semantic_kind")
        record_id = _text(raw_artifact.get("record_id"), field=f"{context}.record_id")
        path = _text(raw_artifact.get("path"), field=f"{context}.path")
        _text(raw_artifact.get("kind"), field=f"{context}.kind")
        _text(raw_artifact.get("scope"), field=f"{context}.scope")
        _text(raw_artifact.get("media_type"), field=f"{context}.media_type")
        if not isinstance(raw_artifact.get("exists"), bool):
            raise ReaderEvidenceManifestAdapterError(f"Reader evidence {context}.exists must be boolean.")
        identity = (semantic_kind, record_id, path)
        if identity in seen:
            raise ReaderEvidenceManifestAdapterError(f"Reader evidence row {row_index} has duplicate artifacts.")
        seen.add(identity)
        artifacts.append(dict(raw_artifact))
    return artifacts


def _count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReaderEvidenceManifestAdapterError(f"Reader evidence {field} must be a non-negative integer.")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ReaderEvidenceManifestAdapterError(f"Reader evidence {field} must be a trimmed, non-empty string.")
    return value


__all__ = [
    "READER_EVIDENCE_API_VERSION",
    "READER_EVIDENCE_MANIFEST_ADAPTER",
    "ReaderEvidenceManifestAdapterError",
    "ReaderEvidenceManifestProjection",
    "parse_reader_evidence_manifest_adapter",
]
