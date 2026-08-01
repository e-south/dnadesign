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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pandas as pd
import yaml

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

from .display_contract import response_example_labels, validate_study_display
from .reader_record_receipt import READER_RESPONSE_RECORD_CONTRACTS
from .reader_record_validation import validate_reader_response_frames

READER_RESPONSE_PROTOCOL_ID = "plate_reader/response_window"
READER_RESPONSE_WINDOW_DIAGNOSTIC_RECORD_ID = "plot:response_window_diagnostic"
STUDY_PROJECTION_SCHEMA = "stress_ethanol_cipro_growth.reader_response_projection.v2"
EXPECTED_RECORDS = READER_RESPONSE_RECORD_CONTRACTS
STATE_ORDER = ("00", "10", "01", "11")
VALUE_COLUMNS = tuple(f"r{state}" for state in STATE_ORDER) + tuple(f"b{state}" for state in STATE_ORDER)

ReaderResponseRecordError = ReaderRecordError
ReaderRecordRef = ReaderResolvedRecord


@dataclass(frozen=True, slots=True)
class ReaderResponseRecords:
    """Verified Reader records plus the study-owned semantic projection."""

    source: ReaderRecordSet
    projection_path: Path
    projection_sha256: str
    projection: Mapping[str, object]
    designs: pd.DataFrame
    descriptive_resampling_draws: pd.DataFrame
    wells: pd.DataFrame
    traces: pd.DataFrame
    events: pd.DataFrame

    def __post_init__(self) -> None:
        frozen = _freeze_contract_value(self.projection)
        object.__setattr__(self, "projection", cast(Mapping[str, object], frozen))

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
    def primary_reduction_id(self) -> str:
        return str(self.projection["primary_reduction_id"])

    @property
    def response_examples(self) -> dict[str, str]:
        return response_example_labels(self.projection["display"])

    @property
    def reference_design_id(self) -> str:
        display = _mapping(self.projection["display"], label="projection.display")
        channels = _mapping(display.get("channels"), label="projection.display.channels")
        return _text(channels.get("reference_design_id"), label="projection.display.channels.reference_design_id")

    def source_receipt(self) -> dict[str, object]:
        receipt = self.source.source_receipt()
        receipt.update(
            {
                "schema_version": STUDY_PROJECTION_SCHEMA,
                "projection_sha256": self.projection_sha256,
            }
        )
        return receipt


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

    projection_file = Path(projection_path).expanduser().resolve()
    projection, projection_sha256 = _load_projection_snapshot(projection_file)
    experiment = Path(experiment_root).expanduser().resolve()
    source = resolve_digest_verified_records(
        experiment / "config.yaml",
        reader_root=reader_root,
        experiment_id=str(projection["reader_experiment_id"]),
        protocol_id=READER_RESPONSE_PROTOCOL_ID,
        expected_records=_record_expectations(),
        reader_command=reader_command,
    )
    frames = {name: _parse_dataframe(record, name=name) for name, record in source.records.items()}
    validate_reader_response_frames(
        designs=frames["designs"],
        draws=frames["descriptive_resampling_draws"],
        wells=frames["wells"],
        traces=frames["traces"],
        events=frames["events"],
        primary_reduction_id=str(projection["primary_reduction_id"]),
        reference_design_id=_projection_reference_design(projection),
    )
    return ReaderResponseRecords(
        source=source,
        projection_path=projection_file,
        projection_sha256=projection_sha256,
        projection=projection,
        designs=frames["designs"],
        descriptive_resampling_draws=frames["descriptive_resampling_draws"],
        wells=frames["wells"],
        traces=frames["traces"],
        events=frames["events"],
    )


def load_reader_response_display_record(
    records: ReaderResponseRecords,
    *,
    reader_command: Sequence[str] | None = None,
) -> ReaderResponseDisplay:
    """Resolve and verify the optional study-pinned diagnostic plot record."""

    specification = _display_artifact_spec(records.projection)
    expectations = {
        **_record_expectations(),
        "diagnostic": ReaderRecordExpectation(
            record_id=READER_RESPONSE_WINDOW_DIAGNOSTIC_RECORD_ID,
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
        "id": "response_window_diagnostic",
        "plugin": "plot/response_window_diagnostic",
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


def _load_projection_snapshot(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise ReaderResponseRecordError(f"study Reader projection is missing: {path}")
    try:
        source_bytes = path.read_bytes()
        payload = yaml.safe_load(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReaderResponseRecordError(f"could not read study Reader projection {path}: {exc}") from exc
    fields = {
        "schema_version",
        "study_id",
        "projection_id",
        "reader_experiment_id",
        "primary_reduction_id",
        "state_order",
        "records",
        "display",
        "display_artifact",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ReaderResponseRecordError(f"study Reader projection fields must be exactly {sorted(fields)}")
    if payload["schema_version"] != STUDY_PROJECTION_SCHEMA:
        raise ReaderResponseRecordError(f"study Reader projection must use {STUDY_PROJECTION_SCHEMA!r}")
    if payload["study_id"] != "stress_ethanol_cipro_growth":
        raise ReaderResponseRecordError("study Reader projection identity disagrees")
    for field_name in ("projection_id", "reader_experiment_id", "primary_reduction_id"):
        _text(payload[field_name], label=f"projection.{field_name}")
    if tuple(payload["state_order"]) != STATE_ORDER:
        raise ReaderResponseRecordError(f"study Reader projection state order must be {STATE_ORDER}")
    configured_records = payload["records"]
    expected_records = {
        name: {"record_id": record_id, "contract_id": contract_id}
        for name, (record_id, contract_id) in EXPECTED_RECORDS.items()
    }
    if configured_records != expected_records:
        raise ReaderResponseRecordError("study Reader projection record contracts disagree with Reader")
    validate_study_display(payload["display"])
    if payload["display_artifact"] is not None:
        _validate_display_artifact_spec(payload["display_artifact"])
    return dict(payload), hashlib.sha256(source_bytes).hexdigest()


def _display_artifact_spec(projection: Mapping[str, object]) -> dict[str, str]:
    value = projection.get("display_artifact")
    if value is None:
        raise ReaderResponseRecordError(
            "study Reader projection has no display_artifact pin; run and verify the canonical Reader "
            "diagnostic, then pin its source experiment, design, producer-config digest, and path"
        )
    return _validate_display_artifact_spec(value)


def _validate_display_artifact_spec(value: object) -> dict[str, str]:
    fields = {"record_id", "source_experiment_id", "design_id", "producer_config_digest", "path"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReaderResponseRecordError(f"projection.display_artifact fields must be exactly {sorted(fields)}")
    result = {field: _text(value[field], label=f"projection.display_artifact.{field}") for field in fields}
    if result["record_id"] != READER_RESPONSE_WINDOW_DIAGNOSTIC_RECORD_ID:
        raise ReaderResponseRecordError(
            f"projection.display_artifact.record_id must be {READER_RESPONSE_WINDOW_DIAGNOSTIC_RECORD_ID!r}"
        )
    _sha256_digest(result["producer_config_digest"], label="projection.display_artifact.producer_config_digest")
    path = Path(result["path"])
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in {".png", ".pdf"}:
        raise ReaderResponseRecordError(
            "projection.display_artifact.path must be a confined outputs-relative PNG or PDF path"
        )
    return result


def _projection_reference_design(projection: Mapping[str, object]) -> str:
    display = _mapping(projection.get("display"), label="projection.display")
    channels = _mapping(display.get("channels"), label="projection.display.channels")
    return _text(channels.get("reference_design_id"), label="projection.display.channels.reference_design_id")


def _freeze_contract_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_contract_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_contract_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_contract_value(item) for item in value)
    return value


def _verify_media_signature(value: ReaderArtifactFile) -> None:
    suffix = Path(value.reader_path).suffix.lower()
    signature = value.content[:8]
    if suffix == ".png" and signature != b"\x89PNG\r\n\x1a\n":
        raise ReaderResponseRecordError("Reader diagnostic PNG signature is invalid")
    if suffix == ".pdf" and not signature.startswith(b"%PDF"):
        raise ReaderResponseRecordError("Reader diagnostic PDF signature is invalid")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReaderResponseRecordError(f"{label} must be an object")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReaderResponseRecordError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256_digest(value: object, *, label: str) -> str:
    token = _text(value, label=label)
    if not token.startswith("sha256:") or len(token) != 71:
        raise ReaderResponseRecordError(f"{label} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in token[7:]):
        raise ReaderResponseRecordError(f"{label} must be a lowercase sha256 digest")
    return token


__all__ = [
    "EXPECTED_RECORDS",
    "READER_CATALOG_SCHEMA_VERSION",
    "READER_RECORD_SCHEMA_VERSION",
    "READER_RESPONSE_PROTOCOL_ID",
    "READER_RESPONSE_WINDOW_DIAGNOSTIC_RECORD_ID",
    "STUDY_PROJECTION_SCHEMA",
    "ReaderRecordRef",
    "ReaderResponseDisplay",
    "ReaderResponseRecordError",
    "ReaderResponseRecords",
    "build_all_primary_measurements",
    "load_reader_response_display_record",
    "load_reader_response_records",
]
