"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_record_receipt.py

Validate the exact Reader record receipt persisted by this study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .artifact_contract import ResponseWindowObservationArtifactError
from .reader_config_attestation import CONFIG_ATTESTATION_SCHEMA
from .reader_projection_contract import READER_EVENT_WINDOW_RECORD_CONTRACTS, STUDY_PROJECTION_SCHEMA
from .reader_record_receipt_records import validate_reader_record_revisions

_RECEIPT_FIELDS = {
    "schema_version",
    "experiment_id",
    "protocol_id",
    "catalog",
    "config",
    "projection_sha256",
    "records",
}
_CATALOG_FIELDS = {"schema_version", "provenance_epoch_id", "sha256"}
_CONFIG_FIELDS = {"schema_version", "config_sha256", "authoring_sha256", "analysis"}


def validate_reader_record_receipt(value: object) -> None:
    """Validate the canonical catalog and record identities persisted downstream."""

    if not isinstance(value, dict):
        raise ResponseWindowObservationArtifactError("Reader record receipt is malformed.")
    if set(value) != _RECEIPT_FIELDS:
        raise ResponseWindowObservationArtifactError("Reader record receipt identity is malformed.")
    catalog = value.get("catalog")
    config = value.get("config")
    records = value.get("records")
    if (
        value.get("schema_version") != STUDY_PROJECTION_SCHEMA
        or not _nonempty_text(value.get("experiment_id"))
        or value.get("protocol_id") != "plate_reader/four_state_event_window"
        or not _is_sha256(value.get("projection_sha256"))
        or not isinstance(catalog, dict)
        or set(catalog) != _CATALOG_FIELDS
        or catalog.get("schema_version") != 4
        or not _is_sha256(catalog.get("sha256"))
        or not _nonempty_text(catalog.get("provenance_epoch_id"))
        or not isinstance(config, dict)
        or set(config) != _CONFIG_FIELDS
        or config.get("schema_version") != CONFIG_ATTESTATION_SCHEMA
        or not _is_sha256(config.get("config_sha256"))
        or not _is_sha256(config.get("authoring_sha256"))
        or not isinstance(config.get("analysis"), dict)
        or not isinstance(records, dict)
    ):
        raise ResponseWindowObservationArtifactError("Reader record receipt identity is malformed.")
    validate_reader_record_revisions(records)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = ["READER_EVENT_WINDOW_RECORD_CONTRACTS", "validate_reader_record_receipt"]
