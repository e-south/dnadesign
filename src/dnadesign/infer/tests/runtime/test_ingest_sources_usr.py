"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_ingest_sources_usr.py

Contracts for infer USR ingest loading and id-subset behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from dnadesign.infer.src.ingest.sources import load_usr_input


def _install_fake_usr_dataset(monkeypatch, records_path: Path) -> None:
    import dnadesign.usr as usr_mod

    class _FakeDataset:
        def __init__(self, dataset_root, dataset_name):
            self.records_path = Path(dataset_root) / dataset_name / "records.parquet"

    monkeypatch.setattr(usr_mod, "Dataset", _FakeDataset)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"PAR1")


def test_load_usr_input_filters_read_table_when_ids_are_provided(monkeypatch, tmp_path: Path) -> None:
    records_path = tmp_path / "demo" / "records.parquet"
    _install_fake_usr_dataset(monkeypatch, records_path)

    calls: list[object] = []

    def _fake_read_table(path, *, columns, filters=None):
        calls.append(filters)
        assert Path(path) == records_path
        assert columns == ["id", "sequence"]
        return pa.table(
            {
                "id": ["id_2", "id_1"],
                "sequence": ["TTTT", "AAAA"],
            }
        )

    monkeypatch.setattr("pyarrow.parquet.read_table", _fake_read_table)

    seqs, ids, _dataset = load_usr_input(
        dataset_name="demo",
        field="sequence",
        root=tmp_path,
        ids=["id_1", "id_missing", "id_2", "id_1"],
    )

    assert len(calls) == 1
    assert isinstance(calls[0], list)
    filter_triplet = calls[0][0]
    assert filter_triplet[0] == "id"
    assert filter_triplet[1] == "in"
    assert set(filter_triplet[2]) == {"id_1", "id_2", "id_missing"}
    assert ids == ["id_1", "id_2", "id_1"]
    assert seqs == ["AAAA", "TTTT", "AAAA"]


def test_load_usr_input_reads_full_table_when_ids_are_omitted(monkeypatch, tmp_path: Path) -> None:
    records_path = tmp_path / "demo" / "records.parquet"
    _install_fake_usr_dataset(monkeypatch, records_path)

    calls: list[object] = []

    def _fake_read_table(path, *, columns, filters=None):
        calls.append(filters)
        assert Path(path) == records_path
        assert columns == ["id", "sequence"]
        return pa.table(
            {
                "id": ["id_1", "id_2"],
                "sequence": ["AAAA", "TTTT"],
            }
        )

    monkeypatch.setattr("pyarrow.parquet.read_table", _fake_read_table)

    seqs, ids, _dataset = load_usr_input(
        dataset_name="demo",
        field="sequence",
        root=tmp_path,
        ids=None,
    )

    assert calls == [None]
    assert ids == ["id_1", "id_2"]
    assert seqs == ["AAAA", "TTTT"]
