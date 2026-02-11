"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_attach_duckdb.py

DuckDB backend attach behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src.overlays import overlay_metadata, overlay_path
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def _row(seq: str, *, source: str = "test") -> dict:
    return {
        "sequence": seq,
        "bio_type": "dna",
        "alphabet": "dna_4",
        "source": source,
    }


def test_attach_duckdb_writes_overlay(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows([_row("ACGT"), _row("TGCA")], source="unit-test")

    attach_path = tmp_path / "attach.parquet"
    tbl = pa.table(
        {
            "id": [ds.head(1).iloc[0]["id"], ds.head(2).iloc[1]["id"]],
            "score": [0.1, 0.2],
        }
    )
    pq.write_table(tbl, attach_path)

    rows = ds.attach(
        attach_path,
        namespace="mock",
        key="id",
        backend="duckdb",
        parse_json=False,
    )
    assert rows == 2

    out_path = overlay_path(ds.dir, "mock")
    assert out_path.exists()
    meta = overlay_metadata(out_path)
    assert meta["namespace"] == "mock"
    assert meta["key"] == "id"

    out_tbl = pq.read_table(out_path)
    assert set(out_tbl.column_names) == {"id", "mock__score"}
