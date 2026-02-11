"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_cli_maintenance_registry.py

CLI maintenance command tests for registry freeze and overlay compaction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.overlays import overlay_dir_path, overlay_path
from dnadesign.usr.src.registry import registry_hash
from dnadesign.usr.tests.registry_helpers import register_test_namespace


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


def test_cli_registry_freeze_creates_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "maintenance", "registry-freeze", "demo"])
    assert result.exit_code == 0

    snap_dir = ds.dir / "_registry"
    assert snap_dir.exists()
    assert list(snap_dir.glob("registry.*.yaml"))

    pf = pq.ParquetFile(str(ds.records_path))
    md = pf.schema_arrow.metadata or {}
    assert md.get(b"usr:registry_hash") == registry_hash(root, required=True).encode("utf-8")


def test_cli_overlay_compact(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    ids = ds.head(2)["id"].tolist()
    tbl = pa.table({"id": ids, "audit__score": [0.1, 0.2]})
    ds.write_overlay_part("audit", tbl, key="id", allow_missing=False)
    ds.write_overlay_part("audit", tbl, key="id", allow_missing=False)

    parts_dir = overlay_dir_path(ds.dir, "audit")
    assert parts_dir.exists()

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--root", str(root), "maintenance", "overlay-compact", "demo", "--namespace", "audit"],
    )
    assert result.exit_code == 0

    assert overlay_path(ds.dir, "audit").exists()
    archived = ds.dir / "_derived" / "_archived" / "audit"
    assert archived.exists()
