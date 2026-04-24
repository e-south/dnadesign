"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/commands/maintenance/test_cli_maintenance_registry.py

CLI maintenance command tests for registry freeze and overlay maintenance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.testsupport.usr import register_test_namespace
from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.overlays import overlay_dir_path, overlay_path
from dnadesign.usr.src.registry import registry_hash


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
    assert not archived.exists()


def test_cli_overlay_remove_archives_overlay(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    ids = ds.head(2)["id"].tolist()
    tbl = pa.table({"id": ids, "audit__score": [0.1, 0.2]})
    ds.write_overlay_part("audit", tbl, key="id", allow_missing=False)

    parts_dir = overlay_dir_path(ds.dir, "audit")
    assert parts_dir.exists()

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--root", str(root), "maintenance", "overlay-remove", "demo", "--namespace", "audit", "--mode", "archive"],
    )
    assert result.exit_code == 0

    archived_root = ds.dir / "_derived" / "_archived"
    assert archived_root.exists()
    assert not parts_dir.exists()
    assert "archived_path" in result.output


def test_cli_overlay_remove_rejects_reserved_namespace(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    ids = ds.head(1)["id"].tolist()
    ds.set_state(ids, masked=True)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "maintenance",
            "overlay-remove",
            "demo",
            "--namespace",
            "usr_state",
            "--mode",
            "archive",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "reserved" in str(result.exception).lower()


def test_cli_overlay_project_projects_namespace_without_touching_other_overlays(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(
        root,
        namespace="densegen",
        columns_spec="densegen__plan:string,densegen__required_regulators:list<string>",
    )
    register_test_namespace(root, namespace="infer", columns_spec="infer__score:float64")

    src = Dataset(root, "densegen_source")
    dest = Dataset(root, "anchor_dest")
    src.init(source="test")
    dest.init(source="test")
    src.import_rows(
        [{"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"}],
        source="unit",
    )
    dest.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "GGGG", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )

    src_id = str(src.head(1, columns=["id"]).iloc[0]["id"])
    dest_rows = dest.head(10, columns=["id", "sequence"]).to_dict(orient="records")
    dest_by_sequence = {str(row["sequence"]): str(row["id"]) for row in dest_rows}

    src.write_overlay(
        "densegen",
        pa.table(
            {
                "id": [src_id],
                "densegen__plan": ["ethanol_f"],
                "densegen__required_regulators": [["cpxR"]],
            }
        ),
        key="id",
        overwrite=True,
    )
    dest.write_overlay(
        "infer",
        pa.table(
            {
                "id": [dest_by_sequence["ACGT"], dest_by_sequence["GGGG"]],
                "infer__score": [0.25, 0.5],
            }
        ),
        key="id",
        overwrite=True,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "maintenance",
            "overlay-project",
            "--src",
            "densegen_source",
            "--dest",
            "anchor_dest",
            "--namespace",
            "densegen",
            "--allow-missing",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "[PROJECTED]" in result.output
    rows = dest.head(10, columns=["sequence", "densegen__plan", "infer__score"]).to_dict(orient="records")
    by_sequence = {str(row["sequence"]): row for row in rows}
    assert by_sequence["ACGT"]["densegen__plan"] == "ethanol_f"
    assert by_sequence["ACGT"]["infer__score"] == 0.25
    assert by_sequence["GGGG"]["infer__score"] == 0.5
