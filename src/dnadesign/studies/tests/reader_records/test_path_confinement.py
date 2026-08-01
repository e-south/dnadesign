"""Path confinement and digest-stability contracts for Reader records."""

from pathlib import Path

import pytest

from dnadesign.studies.core.reader_records import ReaderDataframeRecordError
from dnadesign.studies.core.reader_records import transport as reader_transport

from ._fixtures import _fixture, _page, _reader_runner, _record, _resolve


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
