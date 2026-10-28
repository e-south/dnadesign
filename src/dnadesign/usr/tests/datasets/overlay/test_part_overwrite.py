"""Cell-level overwrite policy for atomic overlay-part writes."""

import json
import multiprocessing
from pathlib import Path

import pyarrow as pa
import pytest

from dnadesign.devtools.tests.support.usr import register_test_namespace
from dnadesign.usr import Dataset, SchemaError


def _dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__a:float64,mock__b:float64")
    ds = Dataset(root, "demo")
    ds.init(source="test")
    ds.import_rows(
        [{"sequence": s, "bio_type": "dna", "alphabet": "dna_4", "source": "test"} for s in ("ACGT", "GGGG")],
        source="test",
    )
    return ds


@pytest.mark.parametrize("storage", ["parts", "compact", "materialized"])
@pytest.mark.parametrize("key", ["id", "sequence", "sequence_ci"])
def test_no_overwrite_preserves_occupied_cells_and_events(tmp_path: Path, storage: str, key: str) -> None:
    ds = _dataset(tmp_path)
    row = ds.head(1).iloc[0]
    value = row["id"] if key == "id" else row["sequence"]
    ds.write_overlay_part("mock", pa.table({key: [value], "mock__a": [1.0]}), key=key)
    if storage == "compact":
        with ds.maintenance(reason="test"):
            ds.compact_overlay("mock")
    elif storage == "materialized":
        with ds.maintenance(reason="test"):
            ds.materialize(keep_overlays=False)
    before = {p.relative_to(ds.dir): p.read_bytes() for p in ds.dir.rglob("*") if p.is_file()}
    with pytest.raises(SchemaError, match="Refusing overwrite"):
        ds.write_overlay_part("mock", pa.table({key: [value], "mock__a": [2.0]}), key=key, overwrite=False)
    assert {p.relative_to(ds.dir): p.read_bytes() for p in ds.dir.rglob("*") if p.is_file()} == before
    ds.write_overlay_part("mock", pa.table({key: [value], "mock__a": [2.0]}), key=key, overwrite=True)
    assert ds.head(2).set_index("id").loc[row["id"], "mock__a"] == 2.0


def test_no_overwrite_allows_null_fill_and_disjoint_cells(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    ids = ds.head(2)["id"].tolist()
    ds.write_overlay_part("mock", pa.table({"id": ids, "mock__a": [1.0, None]}), overwrite=False)
    ds.write_overlay_part("mock", pa.table({"id": [ids[1]], "mock__a": [2.0]}), overwrite=False)
    ds.write_overlay_part("mock", pa.table({"id": [ids[0]], "mock__b": [3.0]}), overwrite=False)
    result = ds.head(2).set_index("id")
    assert result.loc[ids, "mock__a"].tolist() == [1.0, 2.0]
    assert result.loc[ids[0], "mock__b"] == 3.0


def _competing_write(root: str, row_id: str, value: float, barrier, results) -> None:
    ds = Dataset(Path(root), "demo")
    barrier.wait(timeout=15)
    try:
        ds.write_overlay_part("mock", pa.table({"id": [row_id], "mock__a": [value]}), overwrite=False)
    except SchemaError as exc:
        results.put(("rejected", str(exc)))
    else:
        results.put(("written", value))


def test_competing_processes_cannot_both_fill_one_cell(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    row_id = ds.head(1)["id"].iloc[0]
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(2), ctx.Queue()
    processes = [
        ctx.Process(target=_competing_write, args=(str(ds.root), row_id, v, barrier, results)) for v in (1.0, 2.0)
    ]
    try:
        for process in processes:
            process.start()
        outcomes = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert sorted(status for status, _ in outcomes) == ["rejected", "written"]
    winner = next(value for status, value in outcomes if status == "written")
    assert ds.head(2).set_index("id").loc[row_id, "mock__a"] == winner
    events = [json.loads(line) for line in ds.events_path.read_text().splitlines()]
    writes = [event for event in events if event["action"] == "write_overlay_part"]
    assert len(writes) == 1
    assert writes[0]["args"]["overwrite"] is False
