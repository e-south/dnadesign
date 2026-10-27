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
import pytest

from dnadesign.infer.src.errors import ValidationError
from dnadesign.infer.src.ingest.sources import _default_usr_root, load_usr_input, preflight_usr_input


def _install_fake_usr_dataset(monkeypatch, records_path: Path) -> None:
    import dnadesign.usr as usr_mod

    class _FakeDataset:
        def __init__(self, dataset_root, dataset_name):
            self.records_path = Path(dataset_root) / dataset_name / "records.parquet"

    monkeypatch.setattr(usr_mod, "Dataset", _FakeDataset)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_bytes(b"PAR1")


def test_default_usr_root_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DNADESIGN_USR_ROOT", raising=False)
    with pytest.raises(ValidationError, match="USR ingest requires ingest.root or DNADESIGN_USR_ROOT"):
        _default_usr_root()


def test_default_usr_root_uses_env_usr_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DNADESIGN_USR_ROOT", str(tmp_path))
    assert _default_usr_root() == tmp_path.resolve()


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


def test_load_usr_input_requires_explicit_root_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DNADESIGN_USR_ROOT", raising=False)

    with pytest.raises(ValidationError, match="USR ingest requires ingest.root or DNADESIGN_USR_ROOT"):
        load_usr_input(dataset_name="demo", field="sequence", root=None, ids=None)


def test_load_usr_input_normalizes_explicit_usr_package_root(monkeypatch, tmp_path: Path) -> None:
    usr_pkg_root = tmp_path / "usr_pkg"
    usr_pkg_root.mkdir(parents=True, exist_ok=True)
    (usr_pkg_root / "__init__.py").write_text("# test package root\n", encoding="utf-8")
    records_path = usr_pkg_root / "datasets" / "demo" / "records.parquet"
    _install_fake_usr_dataset(monkeypatch, records_path)

    monkeypatch.setattr(
        "pyarrow.parquet.read_table",
        lambda path, *, columns, filters=None: pa.table({"id": ["id_1"], "sequence": ["AAAA"]}),
    )

    seqs, ids, _dataset = load_usr_input(
        dataset_name="demo",
        field="sequence",
        root=usr_pkg_root,
        ids=None,
    )

    assert ids == ["id_1"]
    assert seqs == ["AAAA"]


def test_preflight_usr_input_requires_existing_dataset(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValidationError, match="USR dataset not initialized or missing"):
        preflight_usr_input(dataset_name="missing", field="sequence", root=root)
