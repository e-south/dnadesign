"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_extract_ingest_helper.py

Parity tests for extract ingest loading helper.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dnadesign.infer.src.config import IngestConfig
from dnadesign.infer.src.errors import ConfigError, ValidationError
from dnadesign.infer.src.runtime.ingest_loading import load_extract_ingest


def test_load_extract_ingest_returns_payload_object(monkeypatch) -> None:
    ingest = IngestConfig(source="sequences", field="sequence")
    monkeypatch.setattr("dnadesign.infer.src.runtime.ingest_loading.load_sequences_input", lambda _inputs: ["ACGT", "TGCA"])

    payload = load_extract_ingest(["ACGT", "TGCA"], ingest=ingest)

    assert payload.seqs == ["ACGT", "TGCA"]
    assert payload.ids is None
    assert payload.records is None
    assert payload.pt_path is None
    assert payload.dataset is None


def test_load_extract_ingest_sequences_path(monkeypatch) -> None:
    ingest = IngestConfig(source="sequences", field="sequence")
    monkeypatch.setattr("dnadesign.infer.src.runtime.ingest_loading.load_sequences_input", lambda _inputs: ["ACGT", "TGCA"])

    payload = load_extract_ingest(["ACGT", "TGCA"], ingest=ingest)

    assert payload.seqs == ["ACGT", "TGCA"]
    assert payload.ids is None
    assert payload.records is None
    assert payload.pt_path is None
    assert payload.dataset is None


def test_load_extract_ingest_records_uses_field_fallback(monkeypatch) -> None:
    ingest = SimpleNamespace(source="records", field=None, dataset=None, root=None, ids=None)

    def _load_records(_inputs, field):
        assert field == "sequence"
        return ["ACGT"], [{"sequence": "ACGT"}]

    monkeypatch.setattr("dnadesign.infer.src.runtime.ingest_loading.load_records_input", _load_records)

    payload = load_extract_ingest([{"sequence": "ACGT"}], ingest=ingest)

    assert payload.seqs == ["ACGT"]
    assert payload.records == [{"sequence": "ACGT"}]
    assert payload.ids is None
    assert payload.pt_path is None
    assert payload.dataset is None


def test_load_extract_ingest_pt_file_requires_string_path() -> None:
    ingest = IngestConfig(source="pt_file", field="sequence")

    with pytest.raises(ValidationError, match="inputs must be a path string for pt_file ingest"):
        load_extract_ingest(inputs=["not", "a", "path"], ingest=ingest)


def test_load_extract_ingest_usr_forwards_contract_fields(monkeypatch) -> None:
    ingest = IngestConfig(
        source="usr",
        field="sequence",
        dataset="demo",
        root="/tmp/usr-root",
        ids=["id-1", "id-2"],
    )

    def _load_usr(*, dataset_name, field, root, ids):
        assert dataset_name == "demo"
        assert field == "sequence"
        assert root == "/tmp/usr-root"
        assert ids == ["id-1", "id-2"]
        return ["ACGT"], ["id-1"], SimpleNamespace(name="demo")

    monkeypatch.setattr("dnadesign.infer.src.runtime.ingest_loading.load_usr_input", _load_usr)

    payload = load_extract_ingest(inputs=None, ingest=ingest)

    assert payload.seqs == ["ACGT"]
    assert payload.ids == ["id-1"]
    assert payload.records is None
    assert payload.pt_path is None
    assert getattr(payload.dataset, "name") == "demo"


def test_load_extract_ingest_unknown_source_fails_fast() -> None:
    ingest = SimpleNamespace(source="unknown", field=None, dataset=None, root=None, ids=None)

    with pytest.raises(ConfigError, match="Unknown ingest source"):
        load_extract_ingest(inputs=None, ingest=ingest)
