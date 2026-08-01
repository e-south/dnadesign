"""Bounded pagination contracts for Reader records."""

from copy import deepcopy
from pathlib import Path

import pytest

from dnadesign.studies.core.reader_records import ReaderDataframeRecordError
from dnadesign.studies.core.reader_records import transport as reader_transport

from ._fixtures import _fixture, _page, _record, _resolve, _verify_page


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
