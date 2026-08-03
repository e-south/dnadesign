"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_records.py

Project canonical Reader response-window records into stress-study semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

from dnadesign.studies.core.reader_records import (
    READER_CATALOG_SCHEMA_VERSION,
    READER_RECORD_SCHEMA_VERSION,
    ReaderArtifactFile,
    ReaderRecordError,
    ReaderRecordExpectation,
    ReaderRecordSet,
    ReaderResolvedRecord,
    resolve_digest_verified_records,
)

from .reader_config_attestation import (
    ReaderResponseConfigAttestation,
    attest_reader_response_config,
)
from .reader_projection import (
    READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID,
    READER_EVENT_WINDOW_PROTOCOL_ID,
    READER_EVENT_WINDOW_RECORD_CONTRACTS,
    STATE_ORDER,
    STUDY_PROJECTION_SCHEMA,
    ReaderResponseProjection,
    load_reader_response_projection,
)
from .reader_record_validation import validate_reader_response_frames
from .reader_snapshot import resolve_matching_reader_snapshot

EXPECTED_RECORDS = READER_EVENT_WINDOW_RECORD_CONTRACTS
VALUE_COLUMNS = tuple(f"r{state}" for state in STATE_ORDER) + tuple(f"b{state}" for state in STATE_ORDER)

ReaderResponseRecordError = ReaderRecordError
ReaderRecordRef = ReaderResolvedRecord


@dataclass(frozen=True, slots=True)
class ReaderResponseRecords:
    """Verified Reader records plus the study-owned semantic projection."""

    source: ReaderRecordSet
    projection: ReaderResponseProjection
    config_attestation: ReaderResponseConfigAttestation
    designs: pd.DataFrame
    descriptive_resampling_draws: pd.DataFrame
    wells: pd.DataFrame
    traces: pd.DataFrame
    events: pd.DataFrame
    reader_command: tuple[str, ...] = ()

    @property
    def reader_root(self) -> Path:
        return self.source.reader_root

    @property
    def experiment_root(self) -> Path:
        return self.source.experiment_root

    @property
    def config_path(self) -> Path:
        return self.source.config_path

    @property
    def catalog_path(self) -> Path:
        return self.source.catalog_path

    @property
    def catalog_sha256(self) -> str:
        return self.source.catalog_sha256

    @property
    def provenance_epoch_id(self) -> str:
        return self.source.provenance_epoch_id

    @property
    def experiment_id(self) -> str:
        return self.source.experiment_id

    @property
    def protocol_id(self) -> str:
        return self.source.protocol_id

    @property
    def record_refs(self) -> Mapping[str, ReaderResolvedRecord]:
        return self.source.records

    @property
    def projection_path(self) -> Path:
        return self.projection.path

    @property
    def projection_sha256(self) -> str:
        return self.projection.sha256

    @property
    def primary_reduction_id(self) -> str:
        return self.projection.primary_reduction_id

    @property
    def response_examples(self) -> dict[str, str]:
        return self.projection.response_examples

    @property
    def reference_design_id(self) -> str:
        return self.projection.reference_design_id

    def source_receipt(self) -> dict[str, object]:
        receipt = self.source.source_receipt()
        receipt.update(
            {
                "schema_version": STUDY_PROJECTION_SCHEMA,
                "config": self.config_attestation.to_dict(),
                "projection_sha256": self.projection_sha256,
            }
        )
        return receipt

    def source_receipt_sha256(self) -> str:
        """Return the canonical digest approved by the study policy."""

        encoded = json.dumps(
            self.source_receipt(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def verify_config_attestation(self) -> None:
        """Re-attest the current Reader authoring contract before publication."""

        observed = attest_reader_response_config(
            self.source,
            self.projection,
            reader_command=self.reader_command or None,
        )
        if observed.to_dict() != self.config_attestation.to_dict():
            raise ReaderResponseRecordError("Reader config attestation drifted after record resolution")


@dataclass(frozen=True, slots=True)
class ReaderResponseDisplay:
    """A study-pinned diagnostic from one verified Reader file bundle."""

    source_experiment_id: str
    design_id: str
    record: ReaderResolvedRecord
    selected_file: ReaderArtifactFile

    def to_dict(self) -> dict[str, object]:
        return {
            "source_experiment_id": self.source_experiment_id,
            "design_id": self.design_id,
            "record": self.record.to_dict(),
            "selected_path": self.selected_file.reader_path,
        }


def load_reader_response_records(
    *,
    reader_root: Path,
    experiment_root: Path,
    projection_path: Path,
    reader_command: Sequence[str] | None = None,
) -> ReaderResponseRecords:
    """Resolve the five canonical records and apply study-owned validation."""

    projection = load_reader_response_projection(projection_path)
    experiment = Path(experiment_root).expanduser().resolve()
    source = resolve_digest_verified_records(
        experiment / "config.yaml",
        reader_root=reader_root,
        experiment_id=projection.reader_experiment_id,
        protocol_id=READER_EVENT_WINDOW_PROTOCOL_ID,
        expected_records=_record_expectations(),
        reader_command=reader_command,
    )
    config_attestation = attest_reader_response_config(
        source,
        projection,
        reader_command=reader_command,
    )
    confirmed_source = resolve_matching_reader_snapshot(
        source,
        config_attestation=config_attestation,
        expected_records=_record_expectations(),
        reader_command=reader_command,
        resolver=resolve_digest_verified_records,
    )
    frames = {name: _parse_dataframe(record, name=name) for name, record in confirmed_source.records.items()}
    validate_reader_response_frames(
        designs=frames["designs"],
        draws=frames["descriptive_resampling_draws"],
        wells=frames["wells"],
        traces=frames["traces"],
        events=frames["events"],
        primary_reduction_id=projection.primary_reduction_id,
        reference_design_id=projection.reference_design_id,
        source_experiment_ids=projection.source_experiment_ids,
        event=projection.event,
        aggregation=projection.aggregation,
        reductions=projection.reductions,
    )
    return ReaderResponseRecords(
        source=confirmed_source,
        projection=projection,
        config_attestation=config_attestation,
        designs=frames["designs"],
        descriptive_resampling_draws=frames["descriptive_resampling_draws"],
        wells=frames["wells"],
        traces=frames["traces"],
        events=frames["events"],
        reader_command=tuple(reader_command or ()),
    )


def load_reader_response_display_record(
    records: ReaderResponseRecords,
    *,
    reader_command: Sequence[str] | None = None,
) -> ReaderResponseDisplay:
    """Resolve and verify the optional study-pinned diagnostic plot record."""

    specification = records.projection.display_artifact_spec()
    expectations = {
        **_record_expectations(),
        "diagnostic": ReaderRecordExpectation(
            record_id=READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID,
            kind="file_bundle",
        ),
    }
    source = resolve_digest_verified_records(
        records.config_path,
        reader_root=records.reader_root,
        experiment_id=records.experiment_id,
        protocol_id=records.protocol_id,
        expected_records=expectations,
        reader_command=reader_command,
    )
    if source.catalog_sha256 != records.catalog_sha256 or source.provenance_epoch_id != records.provenance_epoch_id:
        raise ReaderResponseRecordError("Reader catalog changed after response-window records were resolved")
    for name in EXPECTED_RECORDS:
        if source.records[name].to_dict() != records.record_refs[name].to_dict():
            raise ReaderResponseRecordError(f"Reader {name!r} record changed before display resolution")
    diagnostic = source.records["diagnostic"]
    _validate_diagnostic(diagnostic, records=records, specification=specification)
    selected = [item for item in diagnostic.files if item.reader_path == specification["path"]]
    if len(selected) != 1:
        raise ReaderResponseRecordError(
            f"Reader diagnostic pinned path must resolve exactly once: {specification['path']!r}"
        )
    _verify_media_signature(selected[0])
    return ReaderResponseDisplay(
        source_experiment_id=specification["source_experiment_id"],
        design_id=specification["design_id"],
        record=diagnostic,
        selected_file=selected[0],
    )


def build_all_primary_measurements(records: ReaderResponseRecords) -> pd.DataFrame:
    result = records.designs.loc[
        records.designs["reduction_id"].astype(str).eq(records.primary_reduction_id)
        & ~records.designs["is_reference"].astype(bool)
    ].copy()
    result = result.rename(columns={"experiment_id": "reader_experiment_id"})
    result["id"] = result["reader_experiment_id"].astype(str) + "::" + result["design_id"].astype(str)
    return result


def _record_expectations() -> dict[str, ReaderRecordExpectation]:
    return {
        name: ReaderRecordExpectation(record_id=record_id, contract_id=contract_id)
        for name, (record_id, contract_id) in EXPECTED_RECORDS.items()
    }


def _parse_dataframe(record: ReaderResolvedRecord, *, name: str) -> pd.DataFrame:
    if record.kind != "dataframe_artifact" or record.content is None:
        raise ReaderResponseRecordError(f"Reader response-window {name!r} is not a verified dataframe")
    try:
        return pd.read_parquet(BytesIO(record.content))
    except Exception as exc:
        raise ReaderResponseRecordError(f"could not parse Reader response-window {name!r}: {exc}") from exc


def _validate_diagnostic(
    record: ReaderResolvedRecord,
    *,
    records: ReaderResponseRecords,
    specification: Mapping[str, str],
) -> None:
    selection = records.designs.loc[
        records.designs["experiment_id"].astype(str).eq(specification["source_experiment_id"])
        & records.designs["design_id"].astype(str).eq(specification["design_id"])
        & records.designs["reduction_id"].astype(str).eq(records.primary_reduction_id)
    ]
    if len(selection) != 1 or bool(selection.iloc[0]["is_reference"]):
        raise ReaderResponseRecordError(
            "Reader diagnostic study pin must identify exactly one non-reference primary design row"
        )
    expected_producer = {
        "kind": "plot",
        "id": "four_state_event_window_diagnostic",
        "plugin": "plot/four_state_event_window_diagnostic",
    }
    if any(record.producer.get(field) != expected for field, expected in expected_producer.items()):
        raise ReaderResponseRecordError("Reader diagnostic producer identity is invalid")
    if record.producer_config_digest != specification["producer_config_digest"]:
        raise ReaderResponseRecordError("Reader diagnostic producer-config digest disagrees with the study display pin")
    if len(record.inputs) != 2:
        raise ReaderResponseRecordError("Reader diagnostic must consume exactly designs and traces records")
    by_label = {_text(item.get("label"), label="diagnostic input label"): item for item in record.inputs}
    if len(by_label) != len(record.inputs):
        raise ReaderResponseRecordError("Reader diagnostic input labels must be unique")
    expected_inputs = {"designs": records.record_refs["designs"], "traces": records.record_refs["traces"]}
    if set(by_label) != set(expected_inputs):
        raise ReaderResponseRecordError("Reader diagnostic inputs must be exactly designs and traces")
    for label, reference in expected_inputs.items():
        item = by_label[label]
        if (
            item.get("kind") != "record"
            or item.get("discovery_policy") != "record"
            or item.get("record") != reference.record_id
            or item.get("record_revision_digest") != reference.revision_digest
        ):
            raise ReaderResponseRecordError(
                f"Reader diagnostic {label} input does not bind the exact resolved record revision"
            )


def _verify_media_signature(value: ReaderArtifactFile) -> None:
    suffix = Path(value.reader_path).suffix.lower()
    signature = value.content[:8]
    if suffix == ".png" and signature != b"\x89PNG\r\n\x1a\n":
        raise ReaderResponseRecordError("Reader diagnostic PNG signature is invalid")
    if suffix == ".pdf" and not signature.startswith(b"%PDF"):
        raise ReaderResponseRecordError("Reader diagnostic PDF signature is invalid")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReaderResponseRecordError(f"{label} must be a non-empty string")
    return value.strip()


__all__ = [
    "EXPECTED_RECORDS",
    "READER_CATALOG_SCHEMA_VERSION",
    "READER_RECORD_SCHEMA_VERSION",
    "READER_EVENT_WINDOW_PROTOCOL_ID",
    "READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID",
    "STUDY_PROJECTION_SCHEMA",
    "ReaderRecordRef",
    "ReaderResponseDisplay",
    "ReaderResponseRecordError",
    "ReaderResponseRecords",
    "build_all_primary_measurements",
    "load_reader_response_display_record",
    "load_reader_response_records",
]
