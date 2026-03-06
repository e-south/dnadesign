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

from dnadesign.infer.config import IngestConfig
from dnadesign.infer.engine import _load_extract_ingest
from dnadesign.infer.errors import ConfigError, ValidationError


def test_load_extract_ingest_sequences_path(monkeypatch) -> None:
    ingest = IngestConfig(source="sequences", field="sequence")
    monkeypatch.setattr("dnadesign.infer.engine.load_sequences_input", lambda _inputs: ["ACGT", "TGCA"])

    seqs, ids, records, pt_path, ds = _load_extract_ingest(["ACGT", "TGCA"], ingest=ingest)

    assert seqs == ["ACGT", "TGCA"]
    assert ids is None
    assert records is None
    assert pt_path is None
    assert ds is None


def test_load_extract_ingest_records_uses_field_fallback(monkeypatch) -> None:
    ingest = SimpleNamespace(source="records", field=None, dataset=None, root=None, ids=None)

    def _load_records(_inputs, field):
        assert field == "sequence"
        return ["ACGT"], [{"sequence": "ACGT"}]

    monkeypatch.setattr("dnadesign.infer.engine.load_records_input", _load_records)

    seqs, ids, records, pt_path, ds = _load_extract_ingest([{"sequence": "ACGT"}], ingest=ingest)

    assert seqs == ["ACGT"]
    assert records == [{"sequence": "ACGT"}]
    assert ids is None
    assert pt_path is None
    assert ds is None


def test_load_extract_ingest_pt_file_requires_string_path() -> None:
    ingest = IngestConfig(source="pt_file", field="sequence")

    with pytest.raises(ValidationError, match="inputs must be a path string for pt_file ingest"):
        _load_extract_ingest(inputs=["not", "a", "path"], ingest=ingest)


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

    monkeypatch.setattr("dnadesign.infer.engine.load_usr_input", _load_usr)

    seqs, ids, records, pt_path, ds = _load_extract_ingest(inputs=None, ingest=ingest)

    assert seqs == ["ACGT"]
    assert ids == ["id-1"]
    assert records is None
    assert pt_path is None
    assert getattr(ds, "name") == "demo"


def test_load_extract_ingest_unknown_source_fails_fast() -> None:
    ingest = SimpleNamespace(source="unknown", field=None, dataset=None, root=None, ids=None)

    with pytest.raises(ConfigError, match="Unknown ingest source"):
        _load_extract_ingest(inputs=None, ingest=ingest)
