"""Strict codecs for the complete Reader identity used by materialization receipts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from dnadesign.studies.core.reader_records import (
    ReaderRecordError,
    ReaderRecordInputEvidence,
    ReaderRecordProducer,
    parse_record_inputs,
    parse_record_producer,
)

from .._values import MetastudyContractError, _digest, _required_text


@dataclass(frozen=True, slots=True)
class ReaderRecordIdentity:
    """Exact public Reader record identity preserved by one materialization attempt."""

    reader_experiment_id: str
    reader_protocol_id: str
    reader_record_id: str
    reader_record_kind: str
    reader_record_schema_version: int
    reader_record_revision: int
    reader_record_revision_digest: str
    reader_record_config_digest: str
    reader_record_producer_config_digest: str
    reader_record_producer: ReaderRecordProducer
    reader_record_inputs: tuple[ReaderRecordInputEvidence, ...]
    reader_record_contract_id: str
    reader_record_content_digest: str
    reader_record_path: str

    def __post_init__(self) -> None:
        for name in (
            "reader_experiment_id",
            "reader_protocol_id",
            "reader_record_id",
            "reader_record_kind",
            "reader_record_contract_id",
            "reader_record_path",
        ):
            _required_text(getattr(self, name), label=name)
        if self.reader_record_id != "sample_measurements/df":
            raise MetastudyContractError("attempt Reader record_id must equal sample_measurements/df")
        if self.reader_record_kind != "dataframe_artifact":
            raise MetastudyContractError("attempt Reader record kind must equal dataframe_artifact")
        if self.reader_record_schema_version != 6:
            raise MetastudyContractError("attempt Reader record schema version must equal 6")
        if self.reader_record_contract_id != "plate_reader.annotated.v1":
            raise MetastudyContractError("attempt Reader record contract must equal plate_reader.annotated.v1")
        if type(self.reader_record_revision) is not int or self.reader_record_revision < 1:
            raise MetastudyContractError("attempt Reader record revision must be positive")
        _digest(self.reader_record_revision_digest, label="attempt Reader revision digest")
        _digest(self.reader_record_config_digest, label="attempt Reader config digest")
        _digest(self.reader_record_producer_config_digest, label="attempt Reader producer config digest")
        if not isinstance(self.reader_record_producer, ReaderRecordProducer):
            raise MetastudyContractError("attempt Reader producer must be typed public provenance")
        if not isinstance(self.reader_record_inputs, tuple) or not all(
            isinstance(item, ReaderRecordInputEvidence) for item in self.reader_record_inputs
        ):
            raise MetastudyContractError("attempt Reader inputs must be typed public provenance")
        _digest(self.reader_record_content_digest, label="attempt Reader content digest")


def reader_record_identity_payload(identity: ReaderRecordIdentity) -> dict[str, object]:
    """Serialize one complete schema-v6 Reader identity as JSON data."""

    if not isinstance(identity, ReaderRecordIdentity):
        raise MetastudyContractError("Reader record identity must be typed")
    return {
        "reader_experiment_id": identity.reader_experiment_id,
        "reader_protocol_id": identity.reader_protocol_id,
        "reader_record_id": identity.reader_record_id,
        "reader_record_kind": identity.reader_record_kind,
        "reader_record_schema_version": identity.reader_record_schema_version,
        "reader_record_revision": identity.reader_record_revision,
        "reader_record_revision_digest": identity.reader_record_revision_digest,
        "reader_record_config_digest": identity.reader_record_config_digest,
        "reader_record_producer_config_digest": identity.reader_record_producer_config_digest,
        "reader_record_producer": identity.reader_record_producer.to_dict(),
        "reader_record_inputs": [item.to_dict() for item in identity.reader_record_inputs],
        "reader_record_contract_id": identity.reader_record_contract_id,
        "reader_record_content_digest": identity.reader_record_content_digest,
        "reader_record_path": identity.reader_record_path,
    }


def reader_record_identity_from_payload(
    value: Mapping[str, object],
    *,
    index: int,
) -> ReaderRecordIdentity:
    """Parse one strict external identity payload without accepting Python-only shapes."""

    expected = {item.name for item in fields(ReaderRecordIdentity)}
    if set(value) != expected:
        raise MetastudyContractError(f"materialization_attempts[{index}] Reader identity fields changed")
    values = dict(value)
    record_id = values["reader_record_id"]
    if not isinstance(record_id, str):
        raise MetastudyContractError(f"materialization_attempts[{index}] Reader record_id must be text")
    try:
        values["reader_record_producer"] = parse_record_producer(
            values["reader_record_producer"],
            record_id=record_id,
        )
        values["reader_record_inputs"] = parse_record_inputs(
            values["reader_record_inputs"],
            record_id=record_id,
        )
    except ReaderRecordError as exc:
        raise MetastudyContractError(f"materialization_attempts[{index}] Reader lineage is malformed: {exc}") from exc
    return ReaderRecordIdentity(**values)


__all__ = ["ReaderRecordIdentity", "reader_record_identity_from_payload", "reader_record_identity_payload"]
