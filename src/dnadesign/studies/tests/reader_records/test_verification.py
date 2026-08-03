"""Public verification and record-contract validation."""

import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from dnadesign.studies.core.reader_records import (
    ReaderDataframeRecordError,
    ReaderRecordExpectation,
    resolve_digest_verified_records,
)
from dnadesign.studies.core.reader_records import transport as reader_transport

from ._fixtures import _fixture, _page, _reader_runner, _record, _resolve, _verify_page


def test_reader_cli_timeout_fails_closed(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, object] = {}

    def run(command, **kwargs):
        observed.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(reader_transport.subprocess, "run", run)

    with pytest.raises(ReaderDataframeRecordError, match="timed out after 60 seconds"):
        reader_transport.run_reader_json(("reader-fixture", "records"), cwd=tmp_path)
    assert observed["timeout"] == 60


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema="reader.cli/v0"), "reader.cli/v1"),
        (lambda payload: payload.update(command="inspect"), "reader.cli/v1 records payload"),
        (lambda payload: payload["data"]["experiment"].update(evidence=None), "evidence must be an object"),
        (lambda payload: payload["data"]["catalog"].update(schema_version=3), "requires Reader catalog schema v4"),
        (lambda payload: payload["data"]["records"][0].update(schema_version=5), "requires Reader record schema v6"),
        (
            lambda payload: payload["data"]["records"][0].update(content_digest="sha256:" + ("A" * 64)),
            "content_digest must be a lowercase sha256 digest",
        ),
        (
            lambda payload: payload["data"]["records"][0].pop("config_digest"),
            "config_digest must be a non-empty string",
        ),
        (
            lambda payload: payload["data"]["records"][0].pop("producer_config_digest"),
            "producer_config_digest must be a non-empty string",
        ),
        (
            lambda payload: payload["data"]["records"][0].pop("producer"),
            "producer must be an object",
        ),
        (
            lambda payload: payload["data"]["records"][0]["producer"].update(kind="not-a-producer"),
            "producer.kind is unsupported",
        ),
        (
            lambda payload: payload["data"]["records"][0]["inputs"][0].pop("record_revision_digest"),
            "inputs\\[0\\] fields are malformed",
        ),
        (
            lambda payload: payload["data"]["records"][0]["inputs"][0].update(discovery_policy="plugin_discovery"),
            "inputs\\[0\\].discovery_policy must equal 'record'",
        ),
        (lambda payload: payload["data"]["records"][0].update(path="../outside.parquet"), "outputs-relative"),
    ],
)
def test_public_reader_record_rejects_invalid_contracts(tmp_path: Path, monkeypatch, mutation, message: str) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    mutation(payload)
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload))

    with pytest.raises(ReaderDataframeRecordError, match=message):
        _resolve(reader_root, config)


@pytest.mark.parametrize(
    ("verify_payload", "message"),
    [
        (_verify_page(status="failed"), "verify status must be ok"),
        (_verify_page(record_id="other/df"), "did not confirm expected record"),
    ],
)
def test_public_reader_record_requires_successful_public_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verify_payload: dict[str, object],
    message: str,
) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    monkeypatch.setattr(
        reader_transport,
        "run_reader_json",
        _reader_runner(payload, verify_payload=verify_payload),
    )

    with pytest.raises(ReaderDataframeRecordError, match=message):
        _resolve(reader_root, config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["data"]["summary"].update(checked=True),
            "verify summary is malformed",
        ),
        (
            lambda payload: payload["data"]["records"][0].update(kind="file_bundle"),
            "did not confirm expected record",
        ),
        (
            lambda payload: payload["data"].update(issues=[{"code": "synthetic"}]),
            "verify reported issues",
        ),
    ],
)
def test_public_reader_record_rejects_malformed_successful_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    verification = _verify_page()
    mutation(verification)
    monkeypatch.setattr(
        reader_transport,
        "run_reader_json",
        _reader_runner(payload, verify_payload=verification),
    )

    with pytest.raises(ReaderDataframeRecordError, match=message):
        _resolve(reader_root, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revision", None, "revision must be a positive integer"),
        ("revision", 0, "revision must be a positive integer"),
        ("revision", True, "revision must be a positive integer"),
        ("revision_digest", None, "revision_digest must be a non-empty string"),
        ("revision_digest", "sha256:" + ("A" * 64), "revision_digest must be a lowercase sha256 digest"),
    ],
)
def test_public_reader_record_rejects_invalid_exact_revision_identity(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: object,
    message: str,
) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    payload["data"]["records"][0][field] = value
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload))

    with pytest.raises(ReaderDataframeRecordError, match=message):
        _resolve(reader_root, config)


def test_record_set_rejects_inconsistent_experiment_config_digests(tmp_path: Path, monkeypatch) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    primary = _record(digest=digest)
    secondary = _record(digest=digest, record_id="other/df")
    secondary["config_digest"] = "sha256:" + ("e" * 64)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[primary, secondary], total=2)
    verification = _verify_page()
    verification["data"]["summary"]["checked"] = 2
    verification["data"]["records"].append(
        {
            "record_id": "other/df",
            "kind": "dataframe_artifact",
            "schema_version": 6,
            "status": "ok",
            "issues": [],
        }
    )
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload, verify_payload=verification))

    with pytest.raises(ReaderDataframeRecordError, match="must share one experiment config_digest"):
        resolve_digest_verified_records(
            config,
            reader_root=reader_root,
            experiment_id="20260101_demo",
            protocol_id="plate_reader/single_reporter_screen",
            expected_records={
                "primary": ReaderRecordExpectation(
                    record_id="ratio_reporter_normalizer/df",
                    contract_id="plate_reader.annotated.v1",
                ),
                "secondary": ReaderRecordExpectation(
                    record_id="other/df",
                    contract_id="plate_reader.annotated.v1",
                ),
            },
            reader_command=("reader-fixture",),
        )


@pytest.mark.parametrize("identity_kind", ["record_revision", "catalog_epoch"])
def test_public_reader_record_rejects_identity_change_during_resolution(
    tmp_path: Path,
    monkeypatch,
    identity_kind: str,
) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    initial = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    changed = deepcopy(initial)
    if identity_kind == "record_revision":
        changed["data"]["records"][0].update(revision=2, revision_digest="sha256:" + ("b" * 64))
    else:
        changed["data"]["catalog"]["provenance_epoch_id"] = "epoch-changed"
    payloads = iter((initial, changed))

    def run(command, **_kwargs):
        return _verify_page() if "verify" in command else next(payloads)

    monkeypatch.setattr(reader_transport, "run_reader_json", run)

    with pytest.raises(ReaderDataframeRecordError, match="identity changed during resolution"):
        _resolve(reader_root, config)
