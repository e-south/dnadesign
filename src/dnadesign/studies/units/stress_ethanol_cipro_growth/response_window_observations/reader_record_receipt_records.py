"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_record_receipt_records.py

Validate exact record revisions inside a persisted Reader receipt.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import PurePosixPath

from dnadesign.studies.core.reader_records import (
    ReaderRecordError,
    parse_record_inputs,
    parse_record_producer,
)

from .artifact_contract import ResponseWindowObservationArtifactError
from .reader_projection_contract import READER_EVENT_WINDOW_RECORD_CONTRACTS

_RECORD_FIELDS = {
    "record_id",
    "kind",
    "schema_version",
    "revision",
    "revision_digest",
    "config_digest",
    "producer_config_digest",
    "producer",
    "inputs",
    "contract_id",
    "path",
    "size_bytes",
    "content_digest",
}


def validate_reader_record_revisions(value: object) -> None:
    """Validate the five exact, source-closed Reader record revisions."""

    if not isinstance(value, dict) or set(value) != set(READER_EVENT_WINDOW_RECORD_CONTRACTS):
        raise ResponseWindowObservationArtifactError(
            "Reader record receipt must contain exactly the five projected response-window records."
        )
    config_digests: set[str] = set()
    for name, (record_id, contract_id) in READER_EVENT_WINDOW_RECORD_CONTRACTS.items():
        record = value[name]
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise ResponseWindowObservationArtifactError("Reader record receipt contains malformed revisions.")
        if record.get("record_id") != record_id or record.get("contract_id") != contract_id:
            raise ResponseWindowObservationArtifactError(
                f"Reader record receipt {name!r} identity disagrees with the study projection."
            )
        if not _outputs_relative_path(record.get("path")):
            raise ResponseWindowObservationArtifactError(f"Reader record receipt {name!r} path is invalid.")
        if not _valid_revision(record):
            raise ResponseWindowObservationArtifactError("Reader record receipt contains malformed revisions.")
        try:
            parse_record_producer(record.get("producer"), record_id=record_id)
            parse_record_inputs(record.get("inputs"), record_id=record_id)
        except ReaderRecordError as exc:
            raise ResponseWindowObservationArtifactError(
                f"Reader record receipt {name!r} provenance is malformed: {exc}"
            ) from exc
        config_digests.add(str(record["config_digest"]))
    if len(config_digests) != 1:
        raise ResponseWindowObservationArtifactError(
            "Reader record receipt revisions must share one experiment config digest."
        )


def _valid_revision(record: dict[str, object]) -> bool:
    return (
        record.get("kind") == "dataframe_artifact"
        and record.get("schema_version") == 6
        and type(record.get("revision")) is int
        and record["revision"] >= 1
        and _prefixed_sha256(record.get("revision_digest"))
        and _prefixed_sha256(record.get("config_digest"))
        and _prefixed_sha256(record.get("producer_config_digest"))
        and type(record.get("size_bytes")) is int
        and record["size_bytes"] >= 0
        and _prefixed_sha256(record.get("content_digest"))
    )


def _outputs_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path != PurePosixPath(".")
        and ".." not in path.parts
        and path.as_posix() == value
        and not path.parts[0].endswith(":")
    )


def _prefixed_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


__all__ = ["validate_reader_record_revisions"]
