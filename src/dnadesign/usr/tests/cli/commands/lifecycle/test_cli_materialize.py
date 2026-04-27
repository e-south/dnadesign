"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/cli/commands/lifecycle/test_cli_materialize.py

CLI materialize command integration test.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.devtools.testsupport.usr import register_test_namespace
from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.overlays import overlay_path


def _make_dataset(root: Path) -> Dataset:
    register_test_namespace(root, namespace="audit", columns_spec="audit__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    return ds


def test_cli_materialize_keeps_overlays_by_default(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    ids = ds.head(2)["id"].tolist()
    tbl = pa.table({"id": ids, "score": [0.1, 0.2]})
    pq.write_table(tbl, attach_path)

    ds.attach(
        attach_path,
        namespace="audit",
        key="id",
        backend="pyarrow",
        parse_json=False,
    )

    assert overlay_path(ds.dir, "audit").exists()

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "materialize", "demo"])
    assert result.exit_code == 0

    assert overlay_path(ds.dir, "audit").exists()
    schema = pq.ParquetFile(str(ds.records_path)).schema_arrow
    assert "audit__score" in schema.names


def test_cli_materialize_drop_overlays(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    ids = ds.head(2)["id"].tolist()
    tbl = pa.table({"id": ids, "score": [0.1, 0.2]})
    pq.write_table(tbl, attach_path)

    ds.attach(
        attach_path,
        namespace="audit",
        key="id",
        backend="pyarrow",
        parse_json=False,
    )

    assert overlay_path(ds.dir, "audit").exists()

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "materialize", "demo", "--drop-overlays"])
    assert result.exit_code == 0

    assert not overlay_path(ds.dir, "audit").exists()


def test_cli_materialize_archive_overlays_moves_overlay_after_merge(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    ids = ds.head(2)["id"].tolist()
    tbl = pa.table({"id": ids, "score": [0.1, 0.2]})
    pq.write_table(tbl, attach_path)

    ds.attach(
        attach_path,
        namespace="audit",
        key="id",
        backend="pyarrow",
        parse_json=False,
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "materialize", "demo", "--archive-overlays"])
    assert result.exit_code == 0, result.output

    assert not overlay_path(ds.dir, "audit").exists()
    archived = sorted((ds.dir / "_derived" / "_archived").glob("audit-*.parquet"))
    assert len(archived) == 1
    schema = pq.ParquetFile(str(ds.records_path)).schema_arrow
    assert "audit__score" in schema.names
