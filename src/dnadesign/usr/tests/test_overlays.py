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

from dnadesign.usr import Dataset, NamespaceError, SchemaError
from dnadesign.usr.src import dataset as dataset_module
from dnadesign.usr.src.dataset_query import create_overlay_view
from dnadesign.usr.src.duckdb_runtime import connect_duckdb_utc
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


def test_write_overlay_part_after_compact_reopens_parts(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    ids = ds.head(2)["id"].tolist()

    rows1 = ds.write_overlay_part("mock", pa.table({"id": [ids[0]], "mock__score": [1.0]}), key="id")
    assert rows1 == 1

    with ds.maintenance(reason="compact_overlay"):
        compact_path = ds.compact_overlay("mock")
    assert compact_path.exists()

    rows2 = ds.write_overlay_part("mock", pa.table({"id": [ids[1]], "mock__score": [2.0]}), key="id")
    assert rows2 == 1

    assert not compact_path.exists()
    part_dir = ds.dir / "_derived" / "mock"
    parts = sorted(part_dir.glob("part-*.parquet"))
    assert len(parts) == 2

    head = ds.head(2)
    by_id = {row["id"]: row["mock__score"] for row in head[["id", "mock__score"]].to_dict(orient="records")}
    assert by_id[ids[0]] == pytest.approx(1.0)
    assert by_id[ids[1]] == pytest.approx(2.0)


def test_compact_overlay_prunes_older_archives_by_default(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    ids = ds.head(2)["id"].tolist()

    ds.write_overlay_part("mock", pa.table({"id": [ids[0]], "mock__score": [1.0]}), key="id")
    with ds.maintenance(reason="compact_overlay"):
        ds.compact_overlay("mock")

    ds.write_overlay_part("mock", pa.table({"id": [ids[1]], "mock__score": [2.0]}), key="id")
    with ds.maintenance(reason="compact_overlay"):
        ds.compact_overlay("mock")

    archived_root = ds.dir / "_derived" / "_archived" / "mock"
    assert not archived_root.exists()


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


def test_remove_overlay_archive_prunes_older_archives_by_default(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_ids = ds.head(2)["id"].tolist()

    attach_path1 = tmp_path / "attach1.csv"
    pd.DataFrame({"id": [base_ids[0]], "score": [5.0]}).to_csv(attach_path1, index=False)
    ds.attach(attach_path1, namespace="mock", key="id", key_col="id", columns=["score"])
    first = ds.remove_overlay("mock", mode="archive")
    assert Path(first["archived_path"]).exists()

    attach_path2 = tmp_path / "attach2.csv"
    pd.DataFrame({"id": [base_ids[1]], "score": [6.0]}).to_csv(attach_path2, index=False)
    ds.attach(attach_path2, namespace="mock", key="id", key_col="id", columns=["score"])
    second = ds.remove_overlay("mock", mode="archive")
    assert Path(second["archived_path"]).exists()

    archived_root = ds.dir / "_derived" / "_archived"
    snapshots = sorted(path for path in archived_root.glob("mock-*.parquet") if path.exists())
    assert len(snapshots) == 1


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


def test_remove_overlay_rejects_reserved_usr_state_namespace(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    ids = ds.head(1)["id"].tolist()
    ds.set_state(ids, masked=True)

    with pytest.raises(NamespaceError, match="reserved"):
        ds.remove_overlay("usr_state", mode="archive")


def test_write_overlay_part_rejects_reserved_usr_state_namespace(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]

    with pytest.raises(NamespaceError, match="reserved"):
        ds.write_overlay_part(
            "usr_state",
            pa.table({"id": [target_id], "usr_state__masked": [True]}),
            key="id",
        )


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


def test_overlay_read_handles_many_parts_when_duckdb_expression_depth_is_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    counter = {"step": 0}

    def _next_time() -> str:
        base = datetime(2026, 2, 6, tzinfo=timezone.utc)
        value = base + timedelta(seconds=counter["step"])
        counter["step"] += 1
        return value.isoformat()

    monkeypatch.setattr(dataset_module, "now_utc", _next_time)

    for score in range(40):
        ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [float(score)]}), key="id")

    from dnadesign.usr.src import dataset_overlay_query as dataset_overlay_query_module

    connect_default = dataset_overlay_query_module.connect_duckdb_utc

    def _connect_low_depth(*args, **kwargs):
        con = connect_default(*args, **kwargs)
        con.execute("SET max_expression_depth TO 30")
        return con

    monkeypatch.setattr(dataset_overlay_query_module, "connect_duckdb_utc", _connect_low_depth)

    head = ds.head(1)
    assert head.loc[0, "mock__score"] == 39.0


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


def test_overlay_view_construction_scales_without_per_part_temp_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    counter = {"step": 0}

    def _next_time() -> str:
        base = datetime(2026, 2, 6, tzinfo=timezone.utc)
        value = base + timedelta(seconds=counter["step"])
        counter["step"] += 1
        return value.isoformat()

    monkeypatch.setattr(dataset_module, "now_utc", _next_time)

    for score in range(30):
        ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [float(score)]}), key="id")

    overlay_path = ds.dir / "_derived" / "mock"

    class _CountingConnection:
        def __init__(self, inner) -> None:
            self._inner = inner
            self.calls: list[str] = []

        def execute(self, sql: str):
            self.calls.append(sql)
            return self._inner.execute(sql)

    con = connect_duckdb_utc(
        missing_dependency_message="duckdb is required for overlay joins (install duckdb).",
        error_context="overlay scaling test",
    )
    counting_con = _CountingConnection(con)
    create_overlay_view(counting_con, view_name="overlay_perf", path=overlay_path, key="id")
    result = counting_con.execute("SELECT mock__score FROM overlay_perf LIMIT 1").fetchone()

    assert result is not None
    assert float(result[0]) == 29.0
    assert len(counting_con.calls) <= 12


def test_overlay_view_reuses_part_created_at_metadata_across_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    counter = {"step": 0}

    def _next_time() -> str:
        base = datetime(2026, 2, 6, tzinfo=timezone.utc)
        value = base + timedelta(seconds=counter["step"])
        counter["step"] += 1
        return value.isoformat()

    monkeypatch.setattr(dataset_module, "now_utc", _next_time)

    for score in range(5):
        ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [float(score)]}), key="id")

    overlay_path = ds.dir / "_derived" / "mock"
    part_count = len(list(overlay_path.glob("part-*.parquet")))
    assert part_count == 5

    from dnadesign.usr.src import dataset_query as dataset_query_module

    metadata_calls = {"count": 0}
    real_overlay_metadata = dataset_query_module.overlay_metadata

    def _count_overlay_metadata(path: Path):
        metadata_calls["count"] += 1
        return real_overlay_metadata(path)

    monkeypatch.setattr(dataset_query_module, "overlay_metadata", _count_overlay_metadata)

    con = connect_duckdb_utc(
        missing_dependency_message="duckdb is required for overlay joins (install duckdb).",
        error_context="overlay metadata cache test",
    )
    dataset_query_module.create_overlay_view(con, view_name="overlay_cache_a", path=overlay_path, key="id")
    first_call_count = metadata_calls["count"]
    assert first_call_count == part_count

    dataset_query_module.create_overlay_view(con, view_name="overlay_cache_b", path=overlay_path, key="id")
    assert metadata_calls["count"] == first_call_count


def test_overlay_schema_and_metadata_header_reads_are_reused_across_repeated_dataset_reads(
    tmp_path: Path,
) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [1.0]}), key="id")

    from dnadesign.usr.src import overlays as overlays_module

    calls = {"count": 0}
    real_parquet_file = overlays_module.pq.ParquetFile

    def _counted_parquet_file(*args, **kwargs):
        calls["count"] += 1
        return real_parquet_file(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(overlays_module.pq, "ParquetFile", _counted_parquet_file)
        ds.head(5, include_derived=True)
        first_call_count = calls["count"]
        assert first_call_count >= 1
        ds.head(5, include_derived=True)
        second_read_calls = calls["count"] - first_call_count
        assert second_read_calls <= 1


def test_single_file_overlay_duplicate_key_validation_is_reused_across_calls(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [base_id], "score": [3.5]})
    path = tmp_path / "attach.csv"
    df.to_csv(path, index=False)
    ds.attach(path, namespace="mock", key="id", key_col="id", columns=["score"])

    overlay_path = ds.dir / "_derived" / "mock.parquet"
    assert overlay_path.exists()

    class _CountingConnection:
        def __init__(self, inner) -> None:
            self._inner = inner
            self.duplicate_checks = 0

        def execute(self, sql: str):
            if "HAVING COUNT(*) > 1" in sql:
                self.duplicate_checks += 1
            return self._inner.execute(sql)

    con = connect_duckdb_utc(
        missing_dependency_message="duckdb is required for overlay joins (install duckdb).",
        error_context="overlay duplicate cache test",
    )
    counting_con = _CountingConnection(con)
    create_overlay_view(counting_con, view_name="overlay_dup_a", path=overlay_path, key="id")
    create_overlay_view(counting_con, view_name="overlay_dup_b", path=overlay_path, key="id")
    assert counting_con.duplicate_checks == 1


def test_single_file_overlay_uses_inline_relation_without_temp_view(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    base_id = ds.head(1)["id"].iloc[0]
    df = pd.DataFrame({"id": [base_id], "score": [7.5]})
    path = tmp_path / "attach.csv"
    df.to_csv(path, index=False)
    ds.attach(path, namespace="mock", key="id", key_col="id", columns=["score"])

    overlay_path = ds.dir / "_derived" / "mock.parquet"
    assert overlay_path.exists()

    class _CountingConnection:
        def __init__(self, inner) -> None:
            self._inner = inner
            self.temp_view_creates = 0

        def execute(self, sql: str):
            if sql.startswith("CREATE TEMP VIEW overlay_inline"):
                self.temp_view_creates += 1
            return self._inner.execute(sql)

    con = connect_duckdb_utc(
        missing_dependency_message="duckdb is required for overlay joins (install duckdb).",
        error_context="overlay inline test",
    )
    counting_con = _CountingConnection(con)
    source_sql = create_overlay_view(counting_con, view_name="overlay_inline", path=overlay_path, key="id")
    assert source_sql.startswith("read_parquet(")
    assert counting_con.temp_view_creates == 0


def test_multi_part_overlay_duplicate_key_validation_is_reused_across_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    counter = {"step": 0}

    def _next_time() -> str:
        base = datetime(2026, 2, 7, tzinfo=timezone.utc)
        value = base + timedelta(seconds=counter["step"])
        counter["step"] += 1
        return value.isoformat()

    monkeypatch.setattr(dataset_module, "now_utc", _next_time)

    for score in range(5):
        ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [float(score)]}), key="id")

    overlay_path = ds.dir / "_derived" / "mock"
    assert overlay_path.is_dir()

    class _CountingConnection:
        def __init__(self, inner) -> None:
            self._inner = inner
            self.duplicate_checks = 0

        def execute(self, sql: str):
            if "GROUP BY filename" in sql and "HAVING COUNT(*) > 1" in sql:
                self.duplicate_checks += 1
            return self._inner.execute(sql)

    con = connect_duckdb_utc(
        missing_dependency_message="duckdb is required for overlay joins (install duckdb).",
        error_context="overlay duplicate cache test",
    )
    counting_con = _CountingConnection(con)
    create_overlay_view(counting_con, view_name="overlay_mdup_a", path=overlay_path, key="id")
    create_overlay_view(counting_con, view_name="overlay_mdup_b", path=overlay_path, key="id")
    assert counting_con.duplicate_checks == 1


def test_registry_yaml_parse_is_reused_across_repeated_dataset_reads(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [1.0]}), key="id")

    from dnadesign.usr.src import registry as registry_module

    calls = {"count": 0}
    real_safe_load = registry_module.yaml.safe_load

    def _counted_safe_load(*args, **kwargs):
        calls["count"] += 1
        return real_safe_load(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registry_module.yaml, "safe_load", _counted_safe_load)
        registry_module._REGISTRY_CACHE.clear()
        ds.head(5, include_derived=True)
        first_call_count = calls["count"]
        assert first_call_count >= 1
        ds.head(5, include_derived=True)
        assert calls["count"] == first_call_count


def test_list_overlays_reuses_cached_scan_when_dataset_overlay_state_is_unchanged(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [1.0]}), key="id")

    from dnadesign.usr.src import overlays as overlays_module

    calls = {"count": 0}
    real_overlay_parts = overlays_module.overlay_parts

    def _counted_overlay_parts(path):
        calls["count"] += 1
        return real_overlay_parts(path)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(overlays_module, "overlay_parts", _counted_overlay_parts)
        overlays_module.list_overlays(ds.dir)
        first_call_count = calls["count"]
        assert first_call_count >= 1
        overlays_module.list_overlays(ds.dir)
        assert calls["count"] == first_call_count


def test_list_overlays_cache_invalidates_when_new_overlay_part_is_added(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [1.0]}), key="id")

    from dnadesign.usr.src import overlays as overlays_module

    calls = {"count": 0}
    real_overlay_parts = overlays_module.overlay_parts

    def _counted_overlay_parts(path):
        calls["count"] += 1
        return real_overlay_parts(path)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(overlays_module, "overlay_parts", _counted_overlay_parts)
        overlays_module.list_overlays(ds.dir)
        first_call_count = calls["count"]
        ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [2.0]}), key="id")
        overlays_module.list_overlays(ds.dir)
        assert calls["count"] > first_call_count


def test_head_reuses_cached_result_when_dataset_state_is_unchanged(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [1.0]}), key="id")

    calls = {"count": 0}
    real_duckdb_query = ds._duckdb_query

    def _counted_duckdb_query(*args, **kwargs):
        calls["count"] += 1
        return real_duckdb_query(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ds, "_duckdb_query", _counted_duckdb_query)
        df_a = ds.head(5, include_derived=True)
        assert calls["count"] == 1
        df_b = ds.head(5, include_derived=True)
        assert calls["count"] == 1
        assert list(df_a.columns) == list(df_b.columns)
        assert len(df_a) == len(df_b)


def test_head_cache_invalidates_when_overlay_state_changes(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]

    calls = {"count": 0}
    real_duckdb_query = ds._duckdb_query

    def _counted_duckdb_query(*args, **kwargs):
        calls["count"] += 1
        return real_duckdb_query(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ds, "_duckdb_query", _counted_duckdb_query)
        ds.head(5, include_derived=True)
        assert calls["count"] == 1
        ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [2.0]}), key="id")
        ds.head(5, include_derived=True)
        assert calls["count"] == 2


def test_head_pushes_limit_into_overlay_query(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    calls = {"limit": []}
    real_duckdb_query = ds._duckdb_query

    def _counted_duckdb_query(*args, **kwargs):
        calls["limit"].append(kwargs.get("limit"))
        return real_duckdb_query(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ds, "_duckdb_query", _counted_duckdb_query)
        ds.head(5, include_derived=True)

    assert calls["limit"] == [5]


def test_overlay_schema_validation_is_reused_across_repeated_dataset_reads(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [1.0]}), key="id")

    from dnadesign.usr.src import dataset_overlay_catalog as overlay_catalog_module
    from dnadesign.usr.src import dataset_views as dataset_views_module

    calls = {"count": 0}
    real_validate = overlay_catalog_module.validate_overlay_schema

    def _counted_validate(*args, **kwargs):
        calls["count"] += 1
        return real_validate(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(overlay_catalog_module, "validate_overlay_schema", _counted_validate)
        overlay_catalog_module._LOAD_OVERLAYS_CACHE.clear()
        dataset_views_module._HEAD_CACHE.clear()
        ds.head(5, include_derived=True)
        dataset_views_module._HEAD_CACHE.clear()
        ds.head(5, include_derived=True)
    assert calls["count"] == 1


def test_head_state_signature_uses_overlay_paths_without_resolve(tmp_path: Path) -> None:
    from dnadesign.usr.src import dataset_views as dataset_views_module

    class _NoResolvePath:
        st_mtime_ns = 123
        st_size = 456

        def __str__(self) -> str:
            return "/virtual/mock.parquet"

        def stat(self):
            return self

        def resolve(self):
            raise AssertionError("overlay resolve must not be called")

    ds = _make_dataset(tmp_path)
    fake_overlay = _NoResolvePath()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dataset_views_module, "list_overlays", lambda _dataset_dir: [fake_overlay])
        state_sig = dataset_views_module._head_state_signature(ds, include_derived=True)
    assert state_sig[1] == ((str(fake_overlay), 123, 456),)


def test_create_overlay_view_avoids_path_resolve_calls(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    target_id = ds.head(1)["id"].iloc[0]
    ds.write_overlay_part("mock", pa.table({"id": [target_id], "mock__score": [1.0]}), key="id")
    overlay_path = ds.dir / "_derived" / "mock"

    from dnadesign.usr.src import dataset_query as dataset_query_module

    con = connect_duckdb_utc(
        missing_dependency_message="duckdb is required for overlay joins (install duckdb).",
        error_context="overlay resolve test",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            dataset_query_module.Path,
            "resolve",
            lambda self: (_ for _ in ()).throw(AssertionError("resolve should not be called for overlay parts")),
        )
        source = dataset_query_module.create_overlay_view(
            con, view_name="overlay_no_resolve", path=overlay_path, key="id"
        )
    assert str(source)
