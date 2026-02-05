"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_materialize_snapshot_streaming.py

Ensure materialize and snapshot avoid full-table reads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src import dataset as dataset_module
from dnadesign.usr.src.overlays import overlay_path
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def _row(seq: str, *, source: str = "test") -> dict:
    return {
        "sequence": seq,
        "bio_type": "dna",
        "alphabet": "dna_4",
        "source": source,
    }


def test_snapshot_does_not_read_parquet(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows([_row("ACGT")], source="unit-test")

    def _boom(*_args, **_kwargs):
        raise AssertionError("read_parquet should not be called during snapshot")

    monkeypatch.setattr(dataset_module, "read_parquet", _boom)
    ds.snapshot()


def test_materialize_does_not_read_parquet(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows([_row("ACGT"), _row("TGCA")], source="unit-test")

    attach_path = tmp_path / "attach.parquet"
    tbl = pa.table({"id": ds.head(2)["id"].tolist(), "score": [0.1, 0.2]})
    pq.write_table(tbl, attach_path)

    ds.attach(
        attach_path,
        namespace="mock",
        key="id",
        backend="duckdb",
        parse_json=False,
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("read_parquet should not be called during materialize")

    monkeypatch.setattr(dataset_module, "read_parquet", _boom)
    ds.materialize(maintenance=True)

    assert overlay_path(ds.dir, "mock").exists()
