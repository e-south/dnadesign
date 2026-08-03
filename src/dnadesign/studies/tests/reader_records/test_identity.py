"""Identity and replicate metadata contracts for Reader records."""

from pathlib import Path

import pytest

from dnadesign.studies.core.reader_records import (
    ReaderRecordExpectation,
    ReaderRecordInputEvidence,
    ReaderRecordProducer,
    parse_record_inputs,
    resolve_digest_verified_records,
)
from dnadesign.studies.core.reader_records import transport as reader_transport

from ._fixtures import (
    _CONFIG_DIGEST,
    _INPUT_REVISION_DIGEST,
    _PRODUCER_CONFIG_DIGEST,
    _REVISION_DIGEST,
    _fixture,
    _page,
    _reader_runner,
    _record,
    _resolve,
)


def test_public_reader_record_preserves_identity_metadata_and_digest(tmp_path: Path, monkeypatch) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload))

    record = _resolve(reader_root, config)

    assert record.path == artifact.resolve()
    assert record.reader_path == "artifacts/ratio/df.parquet"
    assert record.record_kind == "dataframe_artifact"
    assert record.record_schema_version == 6
    assert record.revision == 1
    assert record.revision_digest == _REVISION_DIGEST
    assert record.config_digest == _CONFIG_DIGEST
    assert record.producer_config_digest == _PRODUCER_CONFIG_DIGEST
    assert isinstance(record.producer, ReaderRecordProducer)
    assert record.producer.to_dict() == {
        "kind": "pipeline",
        "id": "ratio_reporter_normalizer",
        "plugin": "transform/ratio_reporter_normalizer",
        "source_recipe": {
            "recipe": "plate_reader/single_reporter_screen_base",
            "with": {"normalizer_channel": "OD600", "reporter_channel": "RFP"},
        },
    }
    assert len(record.inputs) == 1
    assert isinstance(record.inputs[0], ReaderRecordInputEvidence)
    assert record.inputs[0].to_dict() == {
        "label": "df",
        "kind": "record",
        "record": "labels/df",
        "discovery_policy": "record",
        "record_revision_digest": _INPUT_REVISION_DIGEST,
    }
    assert record.contract_id == "plate_reader.annotated.v1"
    assert record.protocol_id == "plate_reader/single_reporter_screen"
    assert record.replicate_kind == "biological"
    assert record.replicate_identity_field == "biological_replicate_id"


def test_reader_record_set_receipt_preserves_public_provenance(tmp_path: Path, monkeypatch) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload))

    resolved = resolve_digest_verified_records(
        config,
        reader_root=reader_root,
        experiment_id="20260101_demo",
        protocol_id="plate_reader/single_reporter_screen",
        expected_records={
            "dataframe": ReaderRecordExpectation(
                record_id="ratio_reporter_normalizer/df",
                contract_id="plate_reader.annotated.v1",
            )
        },
        reader_command=("reader-fixture",),
    )

    record = resolved.source_receipt()["records"]["dataframe"]
    assert record["config_digest"] == _CONFIG_DIGEST
    assert record["producer_config_digest"] == _PRODUCER_CONFIG_DIGEST
    assert record["producer"]["plugin"] == "transform/ratio_reporter_normalizer"
    assert record["inputs"][0]["record_revision_digest"] == _INPUT_REVISION_DIGEST


@pytest.mark.parametrize(
    "input_evidence",
    (
        {
            "label": "raw",
            "kind": "file",
            "discovery_policy": "plugin_discovery",
            "artifact": {
                "path": "inputs/raw.xlsx",
                "size_bytes": 42,
                "content_digest": "sha256:" + ("e" * 64),
            },
        },
        {
            "label": "source",
            "kind": "source_record",
            "resource": "source_experiment",
            "experiment": "20251231_source",
            "record": "normalized/df",
            "discovery_policy": "source_record",
            "record_revision_digest": "sha256:" + ("f" * 64),
        },
    ),
)
def test_public_reader_input_union_round_trips_without_study_interpretation(
    input_evidence: dict[str, object],
) -> None:
    parsed = parse_record_inputs([input_evidence], record_id="derived/df")

    assert parsed[0].to_dict() == input_evidence


def test_public_reader_record_preserves_unknown_replicate_status_without_inventing_an_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(
        config=config,
        artifact=artifact,
        digest=digest,
        records=[_record(digest=digest)],
        total=1,
        evidence={
            "data_class": "plate_reader_screen",
            "data_class_reason": "fixture",
            "replicate_kind": "unknown",
            "replicate_identity_field": None,
        },
    )
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload))

    record = _resolve(reader_root, config)

    assert record.replicate_kind == "unknown"
    assert record.replicate_identity_field is None


@pytest.mark.parametrize(
    ("replicate_kind", "replicate_identity_field"),
    (("biological", None), ("unknown", "observation_group_id")),
)
def test_public_reader_record_preserves_replicate_scope_without_reinterpreting_it(
    tmp_path: Path,
    monkeypatch,
    replicate_kind: str,
    replicate_identity_field: str | None,
) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(
        config=config,
        artifact=artifact,
        digest=digest,
        records=[_record(digest=digest)],
        total=1,
        evidence={
            "data_class": "plate_reader_screen",
            "data_class_reason": "fixture",
            "replicate_kind": replicate_kind,
            "replicate_identity_field": replicate_identity_field,
        },
    )
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload))

    record = _resolve(reader_root, config)

    assert record.replicate_kind == replicate_kind
    assert record.replicate_identity_field == replicate_identity_field
