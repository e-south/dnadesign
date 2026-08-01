"""Identity and replicate metadata contracts for Reader records."""

from pathlib import Path

import pytest

from dnadesign.studies.core.reader_records import transport as reader_transport

from ._fixtures import _REVISION_DIGEST, _fixture, _page, _reader_runner, _record, _resolve


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
    assert record.contract_id == "plate_reader.annotated.v1"
    assert record.protocol_id == "plate_reader/single_reporter_screen"
    assert record.replicate_kind == "biological"
    assert record.replicate_identity_field == "biological_replicate_id"


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
