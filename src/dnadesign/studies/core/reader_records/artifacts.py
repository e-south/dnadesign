"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/reader_records/artifacts.py

Byte-level resolution for Reader artifacts and file bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from .contracts import (
    READER_RECORD_SCHEMA_VERSION,
    ReaderArtifactFile,
    ReaderRecordExpectation,
    ReaderResolvedRecord,
)
from .provenance import parse_record_inputs, parse_record_producer
from .validation import (
    ReaderRecordError,
    list_value,
    mapping,
    nonnegative_integer,
    positive_revision,
    require_contained,
    sha256_digest,
    text,
)


def resolve_record(
    record: Mapping[str, object],
    *,
    expectation: ReaderRecordExpectation,
    outputs_root: Path,
) -> ReaderResolvedRecord:
    """Resolve and byte-verify one exact record revision."""

    record_id = expectation.record_id
    if record.get("schema_version") != READER_RECORD_SCHEMA_VERSION:
        raise ReaderRecordError(
            f"{record_id}: requires Reader record schema v{READER_RECORD_SCHEMA_VERSION}; "
            "regenerate and verify the Reader experiment"
        )
    kind = text(record.get("kind"), label=f"{record_id}.kind")
    if kind != expectation.kind:
        raise ReaderRecordError(f"{record_id}: kind must equal {expectation.kind!r}")
    revision = positive_revision(record.get("revision"), label=f"{record_id}.revision")
    revision_digest = sha256_digest(record.get("revision_digest"), label=f"{record_id}.revision_digest")
    config_digest = sha256_digest(record.get("config_digest"), label=f"{record_id}.config_digest")
    producer = parse_record_producer(record.get("producer"), record_id=record_id)
    producer_config_digest = sha256_digest(
        record.get("producer_config_digest"),
        label=f"{record_id}.producer_config_digest",
    )
    inputs = parse_record_inputs(record.get("inputs"), record_id=record_id)
    if kind == "dataframe_artifact":
        if record.get("contract_id") != expectation.contract_id:
            raise ReaderRecordError(f"{record_id}: contract must equal {expectation.contract_id!r}")
        reader_path = outputs_relative_path(record.get("path"), label=f"{record_id}.path")
        path = (outputs_root / reader_path).resolve()
        require_contained(path, outputs_root, label=f"Reader record {record_id!r}")
        size_bytes = nonnegative_integer(record.get("size_bytes"), label=f"{record_id}.size_bytes")
        content_digest = sha256_digest(record.get("content_digest"), label=f"{record_id}.content_digest")
        content = verified_bytes(
            path,
            expected_size=size_bytes,
            expected_digest=content_digest,
            label=f"Reader record {record_id!r}",
        )
        return ReaderResolvedRecord._verified(
            record_id=record_id,
            kind=kind,
            schema_version=READER_RECORD_SCHEMA_VERSION,
            revision=revision,
            revision_digest=revision_digest,
            config_digest=config_digest,
            contract_id=expectation.contract_id,
            producer=producer,
            producer_config_digest=producer_config_digest,
            inputs=inputs,
            path=path,
            reader_path=reader_path.as_posix(),
            size_bytes=size_bytes,
            content_digest=content_digest,
            content=content,
            files=(),
        )

    files = resolve_file_evidence(record, outputs_root=outputs_root, record_id=record_id)
    return ReaderResolvedRecord._verified(
        record_id=record_id,
        kind=kind,
        schema_version=READER_RECORD_SCHEMA_VERSION,
        revision=revision,
        revision_digest=revision_digest,
        config_digest=config_digest,
        contract_id=None,
        producer=producer,
        producer_config_digest=producer_config_digest,
        inputs=inputs,
        path=None,
        reader_path=None,
        size_bytes=None,
        content_digest=None,
        content=None,
        files=files,
    )


def resolve_file_evidence(
    record: Mapping[str, object],
    *,
    outputs_root: Path,
    record_id: str,
) -> tuple[ReaderArtifactFile, ...]:
    raw_files = list_value(record.get("files"), label=f"{record_id}.files")
    raw_evidence = list_value(record.get("file_evidence"), label=f"{record_id}.file_evidence")
    if not raw_files:
        raise ReaderRecordError(f"{record_id}: file bundle cannot be empty")
    file_paths = [
        outputs_relative_path(value, label=f"{record_id}.files[{index}]") for index, value in enumerate(raw_files)
    ]
    if len(set(file_paths)) != len(file_paths):
        raise ReaderRecordError(f"{record_id}: file bundle paths must be unique")
    evidence_by_path: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(raw_evidence):
        item = mapping(value, label=f"{record_id}.file_evidence[{index}]")
        if set(item) != {"path", "size_bytes", "content_digest"}:
            raise ReaderRecordError(f"{record_id}: file evidence fields are malformed")
        reader_path = outputs_relative_path(
            item.get("path"),
            label=f"{record_id}.file_evidence[{index}].path",
        ).as_posix()
        if reader_path in evidence_by_path:
            raise ReaderRecordError(f"{record_id}: repeated file evidence path {reader_path!r}")
        evidence_by_path[reader_path] = item
    if {path.as_posix() for path in file_paths} != set(evidence_by_path):
        raise ReaderRecordError(f"{record_id}: files and file evidence disagree")
    result: list[ReaderArtifactFile] = []
    for relative in file_paths:
        reader_path = relative.as_posix()
        evidence = evidence_by_path[reader_path]
        size_bytes = nonnegative_integer(evidence.get("size_bytes"), label=f"{reader_path}.size_bytes")
        content_digest = sha256_digest(evidence.get("content_digest"), label=f"{reader_path}.content_digest")
        path = (outputs_root / relative).resolve()
        require_contained(path, outputs_root, label=f"Reader file {reader_path!r}")
        content = verified_bytes(
            path,
            expected_size=size_bytes,
            expected_digest=content_digest,
            label=f"Reader file {reader_path!r}",
        )
        result.append(
            ReaderArtifactFile(
                reader_path=reader_path,
                path=path,
                size_bytes=size_bytes,
                content_digest=content_digest,
                content=content,
            )
        )
    return tuple(result)


def outputs_relative_path(value: object, *, label: str) -> Path:
    token = Path(text(value, label=label))
    if token.is_absolute() or ".." in token.parts or token == Path("."):
        raise ReaderRecordError(f"{label} must be outputs-relative")
    return token


def verified_bytes(
    path: Path,
    *,
    expected_size: int,
    expected_digest: str,
    label: str,
) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ReaderRecordError(f"could not read {label}: {exc}") from exc
    observed_digest = "sha256:" + hashlib.sha256(content).hexdigest()
    if len(content) != expected_size or observed_digest != expected_digest:
        raise ReaderRecordError(f"{label} byte size or content digest mismatch")
    return content
