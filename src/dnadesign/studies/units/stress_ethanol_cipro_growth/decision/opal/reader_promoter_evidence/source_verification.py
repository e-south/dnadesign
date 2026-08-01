"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/source_verification.py

Verify canonical Reader and study-binding receipts in a display row.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Mapping
from uuid import UUID

from .contracts import READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID, ReaderPromoterEvidenceError
from .display_artifact_verification import is_sha256

_DATAFRAME_FIELDS = {
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
_DIAGNOSTIC_FIELDS = {
    "record_id",
    "kind",
    "schema_version",
    "revision",
    "revision_digest",
    "producer_config_digest",
    "file_evidence",
}


def verify_display_sources(value: object, *, row: Mapping[str, object]) -> dict[str, object]:
    """Verify the exact Reader-record receipt and study binding source."""

    if not isinstance(value, dict) or set(value) != {"response_window", "candidate_bindings"}:
        raise ReaderPromoterEvidenceError("Reader display sources must name response_window and candidate_bindings.")
    response = _verify_response_source(value["response_window"], row=row)
    bindings = value["candidate_bindings"]
    fields = {
        "schema_id",
        "schema_version",
        "study_id",
        "manifest_sha256",
        "records_sha256",
        "candidate_table_id",
        "candidate_selection_sha256",
    }
    if not isinstance(bindings, dict) or set(bindings) != fields:
        raise ReaderPromoterEvidenceError("Reader candidate-binding source fields are malformed.")
    if (
        bindings["schema_id"] != "dnadesign.study.promoter_candidate_bindings.v1"
        or bindings["schema_version"] != "1"
        or bindings["study_id"] != "stress_ethanol_cipro_growth"
        or not all(is_sha256(bindings[field]) for field in fields if field.endswith("sha256"))
    ):
        raise ReaderPromoterEvidenceError("Reader candidate-binding source is invalid.")
    _text(bindings["candidate_table_id"], field="candidate_bindings.candidate_table_id")
    return response


def verify_selected_binding(value: object, *, row: Mapping[str, object]) -> dict[str, object]:
    """Verify one exact Reader-design to study-candidate binding."""

    fields = {
        "reader_design_id",
        "candidate_id",
        "sequence_sha256",
        "sequence_authority_dataset_id",
        "sequence_authority_id",
        "sequence_authority_sha256",
        "source_class",
        "design_family",
        "binding_status",
        "binding_method",
        "densegen_plan",
        "densegen_run_id",
        "densegen_sampling_library_hash",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ReaderPromoterEvidenceError("Reader selected-binding fields are malformed.")
    densegen_fields = ("densegen_plan", "densegen_run_id", "densegen_sampling_library_hash")
    required = fields - set(densegen_fields)
    if (
        any(
            not isinstance(value[field], str) or value[field] != value[field].strip() or not value[field]
            for field in required
        )
        or not is_sha256(value["sequence_sha256"])
        or not is_sha256(value["sequence_authority_sha256"])
        or value["reader_design_id"] != row["design_id"]
        or value["candidate_id"] != row["candidate_id"]
        or value["binding_status"] != "resolved"
        or value["binding_method"] != "exact_alias"
    ):
        raise ReaderPromoterEvidenceError("Reader selected binding is invalid or inconsistent.")
    densegen = tuple(value[field] for field in densegen_fields)
    if value["source_class"] == "densegen":
        if any(not isinstance(item, str) or item != item.strip() or not item for item in densegen):
            raise ReaderPromoterEvidenceError("DenseGen selected binding requires plan, run, and library provenance.")
    elif any(item is not None for item in densegen):
        raise ReaderPromoterEvidenceError("Non-DenseGen selected binding must not carry DenseGen provenance.")
    return dict(value)


def _verify_response_source(value: object, *, row: Mapping[str, object]) -> dict[str, object]:
    fields = {
        "schema_version",
        "output_experiment_id",
        "source_experiment_id",
        "design_id",
        "reduction_id",
        "protocol_id",
        "projection_sha256",
        "catalog",
        "records",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ReaderPromoterEvidenceError("Reader response-window source fields are malformed.")
    if (
        value["schema_version"] != "stress_ethanol_cipro_growth.reader_response_record_source.v1"
        or value["protocol_id"] != "plate_reader/four_state_event_window"
        or value["source_experiment_id"] != row["reader_experiment_id"]
        or value["design_id"] != row["design_id"]
        or value["reduction_id"] != row["reduction_id"]
        or not is_sha256(value["projection_sha256"])
    ):
        raise ReaderPromoterEvidenceError("Reader response-window source identity is inconsistent.")
    _text(value["output_experiment_id"], field="response_window.output_experiment_id")
    catalog = value["catalog"]
    if not isinstance(catalog, dict) or set(catalog) != {"schema_version", "provenance_epoch_id", "sha256"}:
        raise ReaderPromoterEvidenceError("Reader catalog source fields are malformed.")
    if catalog["schema_version"] != 4 or not _uuid4(catalog["provenance_epoch_id"]) or not is_sha256(catalog["sha256"]):
        raise ReaderPromoterEvidenceError("Reader catalog source must be catalog v4 with exact epoch and digest.")
    records = value["records"]
    if not isinstance(records, dict) or set(records) != {"designs", "traces", "diagnostic"}:
        raise ReaderPromoterEvidenceError("Reader response-window source must bind designs, traces, and diagnostic.")
    _verify_dataframe_record(
        records["designs"],
        record_id="four_state_event_window/designs",
        contract_id="plate_reader.four_state_event_window.designs.v4",
    )
    _verify_dataframe_record(
        records["traces"],
        record_id="four_state_event_window/traces",
        contract_id="plate_reader.four_state_event_window.traces.v3",
    )
    _verify_diagnostic_record(records["diagnostic"])
    return dict(value)


def _verify_dataframe_record(value: object, *, record_id: str, contract_id: str) -> None:
    if not isinstance(value, dict) or set(value) != _DATAFRAME_FIELDS:
        raise ReaderPromoterEvidenceError(f"Reader dataframe source {record_id!r} fields are malformed.")
    if (
        value["record_id"] != record_id
        or value["kind"] != "dataframe_artifact"
        or value["schema_version"] != 6
        or value["contract_id"] != contract_id
        or not _positive_int(value["revision"])
        or not is_sha256(value["revision_digest"])
        or not is_sha256(value["content_digest"])
        or not _nonnegative_int(value["size_bytes"])
    ):
        raise ReaderPromoterEvidenceError(f"Reader dataframe source {record_id!r} is invalid.")
    _relative_path(value["path"], field=f"{record_id}.path")


def _verify_diagnostic_record(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _DIAGNOSTIC_FIELDS:
        raise ReaderPromoterEvidenceError("Reader diagnostic source fields are malformed.")
    if (
        value["record_id"] != READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID
        or value["kind"] != "file_bundle"
        or value["schema_version"] != 6
        or not _positive_int(value["revision"])
        or not is_sha256(value["revision_digest"])
        or not is_sha256(value["producer_config_digest"])
    ):
        raise ReaderPromoterEvidenceError("Reader diagnostic source is invalid.")
    evidence = value["file_evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ReaderPromoterEvidenceError("Reader diagnostic source has no file evidence.")
    paths: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "content_digest"}:
            raise ReaderPromoterEvidenceError("Reader diagnostic file evidence is malformed.")
        path = _relative_path(item["path"], field="diagnostic.file_evidence.path")
        if path in paths or not _nonnegative_int(item["size_bytes"]) or not is_sha256(item["content_digest"]):
            raise ReaderPromoterEvidenceError("Reader diagnostic file evidence is invalid.")
        paths.add(path)


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ReaderPromoterEvidenceError(f"Reader display {field} must be trimmed non-empty text.")
    return value


def _relative_path(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ReaderPromoterEvidenceError(f"Reader display {field} must be a confined relative path.")
    return text


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def _nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _uuid4(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


__all__ = ["verify_display_sources", "verify_selected_binding"]
