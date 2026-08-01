"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_record_receipt.py

Validate the exact Reader record receipt persisted by this study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import PurePosixPath

from .artifact_contract import ResponseWindowObservationArtifactError

READER_EVENT_WINDOW_RECORD_CONTRACTS = {
    "wells": ("four_state_event_window/wells", "plate_reader.four_state_event_window.wells.v3"),
    "designs": ("four_state_event_window/designs", "plate_reader.four_state_event_window.designs.v4"),
    "descriptive_resampling_draws": (
        "four_state_event_window/descriptive_resampling_draws",
        "plate_reader.four_state_event_window.descriptive_resampling_draws.v3",
    ),
    "traces": ("four_state_event_window/traces", "plate_reader.four_state_event_window.traces.v3"),
    "events": ("four_state_event_window/events", "plate_reader.four_state_event_window.events.v2"),
}
_RECORD_FIELDS = {
    "record_id",
    "kind",
    "schema_version",
    "revision",
    "revision_digest",
    "contract_id",
    "path",
    "size_bytes",
    "content_digest",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "experiment_id",
    "protocol_id",
    "catalog",
    "projection_sha256",
    "records",
}
_CATALOG_FIELDS = {"schema_version", "provenance_epoch_id", "sha256"}


def validate_reader_record_receipt(value: object) -> None:
    """Validate the canonical catalog and record identities persisted downstream."""

    if not isinstance(value, dict):
        raise ResponseWindowObservationArtifactError("Reader record receipt is malformed.")
    if set(value) != _RECEIPT_FIELDS:
        raise ResponseWindowObservationArtifactError("Reader record receipt identity is malformed.")
    catalog = value.get("catalog")
    records = value.get("records")
    if (
        value.get("schema_version") != "stress_ethanol_cipro_growth.reader_response_projection.v3"
        or not _nonempty_text(value.get("experiment_id"))
        or value.get("protocol_id") != "plate_reader/four_state_event_window"
        or not _is_sha256(value.get("projection_sha256"))
        or not isinstance(catalog, dict)
        or set(catalog) != _CATALOG_FIELDS
        or catalog.get("schema_version") != 4
        or not _is_sha256(catalog.get("sha256"))
        or not _nonempty_text(catalog.get("provenance_epoch_id"))
        or not isinstance(records, dict)
    ):
        raise ResponseWindowObservationArtifactError("Reader record receipt identity is malformed.")
    if set(records) != set(READER_EVENT_WINDOW_RECORD_CONTRACTS):
        raise ResponseWindowObservationArtifactError(
            "Reader record receipt must contain exactly the five projected response-window records."
        )
    for name, (record_id, contract_id) in READER_EVENT_WINDOW_RECORD_CONTRACTS.items():
        record = records[name]
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise ResponseWindowObservationArtifactError("Reader record receipt contains malformed revisions.")
        if record.get("record_id") != record_id or record.get("contract_id") != contract_id:
            raise ResponseWindowObservationArtifactError(
                f"Reader record receipt {name!r} identity disagrees with the study projection."
            )
        if not _outputs_relative_path(record.get("path")):
            raise ResponseWindowObservationArtifactError(f"Reader record receipt {name!r} path is invalid.")
        if (
            record.get("kind") != "dataframe_artifact"
            or record.get("schema_version") != 6
            or type(record.get("revision")) is not int
            or record["revision"] < 1
            or not _prefixed_sha256(record.get("revision_digest"))
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] < 0
            or not _prefixed_sha256(record.get("content_digest"))
        ):
            raise ResponseWindowObservationArtifactError("Reader record receipt contains malformed revisions.")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


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


__all__ = ["READER_EVENT_WINDOW_RECORD_CONTRACTS", "validate_reader_record_receipt"]
