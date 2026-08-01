"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_reader_records.py

Boundary tests for Reader's public record handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from dnadesign.studies.core.reader_records import (
    ReaderDataframeRecordError,
    resolve_digest_verified_dataframe_record,
)
from dnadesign.studies.core.reader_records import transport as reader_transport

_REVISION_DIGEST = "sha256:" + ("a" * 64)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    reader_root = tmp_path / "reader"
    experiment = reader_root / "experiments" / "2026" / "20260101_demo"
    config = experiment / "config.yaml"
    artifact = experiment / "outputs" / "artifacts" / "ratio" / "df.parquet"
    artifact.parent.mkdir(parents=True)
    config.write_text("fixture", encoding="utf-8")
    artifact.write_bytes(b"parquet fixture bytes")
    catalog = experiment / "outputs" / "manifests" / "records.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}", encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    return reader_root, config, artifact, digest


def _page(
    *,
    config: Path,
    artifact: Path,
    digest: str,
    records: list[dict[str, object]],
    total: int,
    truncated: bool = False,
    continuation: str | None = None,
    schema: str = "reader.cli/v1",
    evidence: object = ...,
) -> dict[str, object]:
    experiment = config.parent
    return {
        "schema": schema,
        "ok": True,
        "command": "records",
        "data": {
            "experiment": {
                "id": "20260101_demo",
                "title": "fixture",
                "lifecycle": "active",
                "protocol": "plate_reader/single_reporter_screen",
                "config": str(config),
                "root": str(experiment),
                "evidence": {
                    "data_class": "plate_reader_screen",
                    "data_class_reason": "fixture",
                    "replicate_kind": "biological",
                    "replicate_identity_field": "biological_replicate_id",
                }
                if evidence is ...
                else evidence,
            },
            "catalog": {
                "path": str(experiment / "outputs" / "manifests" / "records.json"),
                "outputs_root": str(experiment / "outputs"),
                "schema_version": 4,
                "provenance_epoch_id": "epoch-fixture",
                "active_invocation_ledger": str(
                    experiment / "outputs" / "manifests" / "invocations" / "epoch-fixture.jsonl"
                ),
            },
            "selection": {"include_history": False},
            "summary": {"records": total, "history": {"included": False, "revisions": None}},
            "records": records,
        },
        "error": None,
        "meta": {"projection": "full", "truncated": truncated, "continuation": continuation},
    }


def _record(
    *, digest: str, path: str = "artifacts/ratio/df.parquet", record_id: str = "ratio_reporter_normalizer/df"
) -> dict[str, object]:
    return {
        "schema_version": 6,
        "record_id": record_id,
        "kind": "dataframe_artifact",
        "contract_id": "plate_reader.annotated.v1",
        "content_digest": digest,
        "size_bytes": len(b"parquet fixture bytes"),
        "path": path,
        "revision": 1,
        "revision_digest": _REVISION_DIGEST,
    }


def _verify_page(
    *,
    status: str = "ok",
    record_id: str = "ratio_reporter_normalizer/df",
) -> dict[str, object]:
    record_status = "ok" if status == "ok" else status
    return {
        "schema": "reader.cli/v1",
        "ok": True,
        "command": "verify",
        "data": {
            "schema": "reader.verify/v1",
            "status": status,
            "summary": {
                "checked": 1,
                "failed": 0 if status == "ok" else 1,
                "unverifiable": 0,
                "invocations_checked": 1,
                "invocation_failures": 0,
            },
            "issues": [],
            "records": [
                {
                    "record_id": record_id,
                    "kind": "dataframe_artifact",
                    "schema_version": 6,
                    "status": record_status,
                    "issues": [],
                }
            ],
        },
        "error": None,
        "meta": {"projection": "full", "truncated": False, "continuation": None},
    }


def _reader_runner(
    records_payload: dict[str, object],
    *,
    verify_payload: dict[str, object] | None = None,
):
    verification = _verify_page() if verify_payload is None else verify_payload

    def run(command, **_kwargs):
        return verification if "verify" in command else records_payload

    return run


def _resolve(reader_root: Path, config: Path):
    return resolve_digest_verified_dataframe_record(
        config,
        reader_root=reader_root,
        experiment_id="20260101_demo",
        protocol_id="plate_reader/single_reporter_screen",
        record_id="ratio_reporter_normalizer/df",
        contract_id="plate_reader.annotated.v1",
        reader_command=("reader-fixture",),
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


def test_public_reader_record_follows_bounded_pages(tmp_path: Path, monkeypatch) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    first_page = _page(
        config=config,
        artifact=artifact,
        digest=digest,
        records=[_record(digest=digest, record_id="a/df")],
        total=2,
        truncated=True,
        continuation="opaque",
    )
    final_page = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=2)
    pages = iter([first_page, final_page, deepcopy(first_page), deepcopy(final_page)])
    commands: list[tuple[str, ...]] = []

    def run(command, **_kwargs):
        commands.append(tuple(command))
        return _verify_page() if "verify" in command else next(pages)

    monkeypatch.setattr(reader_transport, "run_reader_json", run)

    assert _resolve(reader_root, config).record_id == "ratio_reporter_normalizer/df"
    assert "--continuation" not in commands[0]
    assert commands[1][-2:] == ("--continuation", "opaque")
    assert commands[2][-4:] == ("verify", str(config), "--format", "json")
    assert "--continuation" not in commands[3]
    assert commands[4][-2:] == ("--continuation", "opaque")


def test_public_reader_record_rejects_repeated_continuation_token(tmp_path: Path, monkeypatch) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    first_page = _page(
        config=config,
        artifact=artifact,
        digest=digest,
        records=[_record(digest=digest, record_id="a/df")],
        total=2,
        truncated=True,
        continuation="opaque",
    )
    repeated_page = _page(
        config=config,
        artifact=artifact,
        digest=digest,
        records=[_record(digest=digest)],
        total=2,
        truncated=True,
        continuation="opaque",
    )
    pages = iter((first_page, repeated_page))
    monkeypatch.setattr(reader_transport, "run_reader_json", lambda *_args, **_kwargs: next(pages))

    with pytest.raises(ReaderDataframeRecordError, match="repeated continuation token"):
        _resolve(reader_root, config)


def test_public_reader_record_rejects_truncated_empty_page(tmp_path: Path, monkeypatch) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(
        config=config,
        artifact=artifact,
        digest=digest,
        records=[],
        total=1,
        truncated=True,
        continuation="opaque",
    )
    monkeypatch.setattr(reader_transport, "run_reader_json", lambda *_args, **_kwargs: payload)

    with pytest.raises(ReaderDataframeRecordError, match="truncated page must contain at least one record"):
        _resolve(reader_root, config)


def test_public_reader_record_rejects_pagination_beyond_page_bound(tmp_path: Path, monkeypatch) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(
        config=config,
        artifact=artifact,
        digest=digest,
        records=[_record(digest=digest, record_id="a/df")],
        total=2,
        truncated=True,
        continuation="opaque",
    )
    calls = 0

    def run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return payload

    monkeypatch.setattr(reader_transport, "MAX_RECORD_PAGES", 1)
    monkeypatch.setattr(reader_transport, "run_reader_json", run)

    with pytest.raises(ReaderDataframeRecordError, match="exceeded the 1-page safety bound"):
        _resolve(reader_root, config)
    assert calls == 1


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


def test_public_reader_record_rejects_digest_drift(tmp_path: Path, monkeypatch) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload))
    artifact.write_bytes(b"drift")

    with pytest.raises(ReaderDataframeRecordError, match="content digest mismatch"):
        _resolve(reader_root, config)


def test_public_reader_record_rejects_input_config_outside_reader_root(tmp_path: Path, monkeypatch) -> None:
    reader_root, _config, _artifact, _digest = _fixture(tmp_path)
    outside_config = tmp_path / "outside" / "config.yaml"
    outside_config.parent.mkdir()
    outside_config.write_text("fixture", encoding="utf-8")

    def unexpected_cli_call(*_args, **_kwargs):
        raise AssertionError("Reader CLI must not run for an out-of-root config")

    monkeypatch.setattr(reader_transport, "run_reader_json", unexpected_cli_call)

    with pytest.raises(ReaderDataframeRecordError, match="Reader config escapes"):
        _resolve(reader_root, outside_config)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("config", "Reader CLI config escapes"),
        ("root", "Reader experiment root escapes"),
        ("outputs_root", "Reader outputs root escapes"),
        ("manifest", "Reader record manifest escapes"),
    ],
)
def test_public_reader_record_rejects_public_paths_outside_reader_root(
    tmp_path: Path,
    monkeypatch,
    field: str,
    message: str,
) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    outside = tmp_path / "outside" / field
    experiment = payload["data"]["experiment"]
    catalog = payload["data"]["catalog"]
    if field in {"config", "root"}:
        experiment[field] = str(outside)
    elif field == "outputs_root":
        catalog["outputs_root"] = str(outside)
    else:
        catalog["path"] = str(outside)
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload))

    with pytest.raises(ReaderDataframeRecordError, match=message):
        _resolve(reader_root, config)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("root", "Reader experiment root must equal the config parent"),
        ("outputs_root", "Reader outputs root must equal the experiment outputs directory"),
        ("manifest", "Reader record manifest must equal the canonical records manifest"),
    ],
)
def test_public_reader_record_rejects_noncanonical_public_paths(
    tmp_path: Path,
    monkeypatch,
    field: str,
    message: str,
) -> None:
    reader_root, config, artifact, digest = _fixture(tmp_path)
    payload = _page(config=config, artifact=artifact, digest=digest, records=[_record(digest=digest)], total=1)
    experiment = payload["data"]["experiment"]
    catalog = payload["data"]["catalog"]
    if field == "root":
        experiment["root"] = str(config.parent.parent)
    elif field == "outputs_root":
        catalog["outputs_root"] = str(config.parent / "alternate_outputs")
    else:
        catalog["path"] = str(config.parent / "outputs" / "manifests" / "alternate.json")
    monkeypatch.setattr(reader_transport, "run_reader_json", _reader_runner(payload))

    with pytest.raises(ReaderDataframeRecordError, match=message):
        _resolve(reader_root, config)
