"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_overlays.py

Tests overlay-first attach and materialize behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.usr import Dataset, SchemaError
from dnadesign.usr.src import dataset as dataset_module
from dnadesign.usr.src.overlays import OVERLAY_META_CREATED, with_overlay_metadata
from dnadesign.usr.src.storage.parquet import now_utc
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def _make_dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "test"},
            {"sequence": "GGGG", "bio_type": "dna", "alphabet": "dna_4", "source": "test"},
        ],
        source="test",
    )
    return ds


def test_attach_creates_overlay_and_head_includes_derived(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [base_id], "score": [3.5]})
    path = tmp_path / "attach.csv"
    df.to_csv(path, index=False)

    ds.attach(path, namespace="mock", key="id", key_col="id", columns=["score"])

    overlay_path = ds.dir / "_derived" / "mock.parquet"
    assert overlay_path.exists()

    base_cols = pq.ParquetFile(str(ds.records_path)).schema_arrow.names
    assert "mock__score" not in base_cols

    head = ds.head(1)
    assert "mock__score" in head.columns
    assert head["mock__score"].iloc[0] == 3.5


def test_materialize_folds_overlays(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [base_id], "score": [1.0]})
    path = tmp_path / "attach.csv"
    df.to_csv(path, index=False)

    ds.attach(path, namespace="mock", key="id", key_col="id", columns=["score"])
    with pytest.raises(SchemaError):
        ds.materialize()
    with ds.maintenance(reason="materialize"):
        ds.materialize(keep_overlays=False)

    base_cols = pq.ParquetFile(str(ds.records_path)).schema_arrow.names
    assert "mock__score" in base_cols
    derived_dir = ds.dir / "_derived"
    assert not any(derived_dir.glob("*.parquet"))


def test_head_succeeds_after_materialize_with_kept_overlays(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [base_id], "score": [4.25]})
    path = tmp_path / "attach.csv"
    df.to_csv(path, index=False)

    ds.attach(path, namespace="mock", key="id", key_col="id", columns=["score"])
    with ds.maintenance(reason="materialize"):
        ds.materialize(keep_overlays=True)

    head = ds.head(1)
    assert "mock__score" in head.columns
    assert list(head.columns).count("mock__score") == 1
    assert head["mock__score"].iloc[0] == 4.25


def test_info_and_schema_after_materialize_with_kept_overlays(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [base_id], "score": [2.5]})
    path = tmp_path / "attach.csv"
    df.to_csv(path, index=False)

    ds.attach(path, namespace="mock", key="id", key_col="id", columns=["score"])
    with ds.maintenance(reason="materialize"):
        ds.materialize(keep_overlays=True)

    info = ds.info()
    assert list(info.columns).count("mock__score") == 1
    assert "mock__score" in ds.schema().names


def test_write_overlay_part_and_compact(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    ids = ds.head(2)["id"].tolist()

    tbl1 = pa.table({"id": [ids[0]], "mock__score": [1.0]})
    rows1 = ds.write_overlay_part("mock", tbl1, key="id")
    assert rows1 == 1

    tbl2 = pa.table({"id": [ids[1]], "mock__score": [2.0]})
    rows2 = ds.write_overlay_part("mock", tbl2, key="id")
    assert rows2 == 1

    part_dir = ds.dir / "_derived" / "mock"
    parts = list(part_dir.glob("part-*.parquet"))
    assert len(parts) == 2

    head = ds.head(2)
    assert "mock__score" in head.columns

    with pytest.raises(SchemaError):
        ds.compact_overlay("mock")

    with ds.maintenance(reason="compact_overlay"):
        compact_path = ds.compact_overlay("mock")
    assert compact_path.exists()
    assert not part_dir.exists()


def test_remove_overlay_archive_moves_overlay_file_and_logs_event(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_id = ds.head(1)["id"].iloc[0]
    attach_path = tmp_path / "attach.csv"
    pd.DataFrame({"id": [base_id], "score": [5.0]}).to_csv(attach_path, index=False)
    ds.attach(attach_path, namespace="mock", key="id", key_col="id", columns=["score"])

    overlay_path = ds.dir / "_derived" / "mock.parquet"
    assert overlay_path.exists()

    result = ds.remove_overlay("mock", mode="archive")
    assert result["removed"] is True
    assert result["namespace"] == "mock"
    archived_path = Path(result["archived_path"])
    assert archived_path.exists()
    assert not overlay_path.exists()

    rows = [json.loads(line) for line in ds.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row.get("action") == "archive_overlay" for row in rows)


def test_remove_overlay_delete_removes_overlay_parts_directory(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    ids = ds.head(2)["id"].tolist()
    ds.write_overlay_part("mock", pa.table({"id": [ids[0]], "mock__score": [1.0]}), key="id")

    part_dir = ds.dir / "_derived" / "mock"
    assert part_dir.exists()

    result = ds.remove_overlay("mock", mode="delete")
    assert result == {"removed": True, "namespace": "mock"}
    assert not part_dir.exists()

    rows = [json.loads(line) for line in ds.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row.get("action") == "remove_overlay" for row in rows)


def test_remove_overlay_rejects_conflicting_file_and_parts_sources(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    ids = ds.head(2)["id"].tolist()
    overlay_file = ds.dir / "_derived" / "mock.parquet"
    overlay_dir = ds.dir / "_derived" / "mock"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    part_path = overlay_dir / "part-0001.parquet"

    table = pa.table({"id": [ids[0]], "mock__score": [3.0]})
    table = with_overlay_metadata(table, namespace="mock", key="id", created_at=now_utc())
    pq.write_table(table, overlay_file)
    pq.write_table(table, part_path)

    with pytest.raises(SchemaError, match="both file and directory sources"):
        ds.remove_overlay("mock", mode="delete")


def test_overlay_parts_last_writer_wins_on_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    counter = {"step": 0}

    def _next_time() -> str:
        base = datetime(2026, 2, 6, tzinfo=timezone.utc)
        value = base + timedelta(seconds=counter["step"])
        counter["step"] += 1
        return value.isoformat()

    monkeypatch.setattr(dataset_module, "now_utc", _next_time)

    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [1.0]}), key="id")
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [9.0]}), key="id")

    head = ds.head(1)
    assert head.loc[0, "mock__score"] == 9.0


def test_overlay_parts_last_writer_wins_on_materialize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    counter = {"step": 0}

    def _next_time() -> str:
        base = datetime(2026, 2, 6, tzinfo=timezone.utc)
        value = base + timedelta(seconds=counter["step"])
        counter["step"] += 1
        return value.isoformat()

    monkeypatch.setattr(dataset_module, "now_utc", _next_time)

    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [2.0]}), key="id")
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [7.0]}), key="id")

    with ds.maintenance(reason="materialize"):
        ds.materialize(namespaces=["mock"], keep_overlays=False)

    materialized = ds.head(1, include_derived=False)
    assert materialized.loc[0, "mock__score"] == 7.0


def test_overlay_parts_require_created_at_metadata(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [1.0]}), key="id")

    part_dir = ds.dir / "_derived" / "mock"
    part = sorted(part_dir.glob("part-*.parquet"))[0]
    table = pq.read_table(part)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(OVERLAY_META_CREATED.encode("utf-8"), None)
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, part)

    with pytest.raises(SchemaError, match="missing required created_at metadata"):
        ds.head(1)
