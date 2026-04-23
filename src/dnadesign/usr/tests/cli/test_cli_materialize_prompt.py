"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_cli_materialize_prompt.py

Materialize CLI prompt behavior tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.usr.src import cli as cli_module
from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.overlays import overlay_path
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


def _attach_overlay(ds: Dataset, path: Path) -> None:
    tbl = pa.table({"id": ds.head(2)["id"].tolist(), "score": [0.1, 0.2]})
    pq.write_table(tbl, path)
    ds.attach(
        path,
        namespace="audit",
        key="id",
        backend="pyarrow",
        parse_json=False,
    )


def test_materialize_prompts_and_aborts(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    _attach_overlay(ds, attach_path)

    monkeypatch.setattr(cli_module, "_is_interactive", lambda: True)
    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "materialize", "demo"], input="n\n")

    assert result.exit_code != 0
    assert overlay_path(ds.dir, "audit").exists()


def test_materialize_yes_bypasses_prompt(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    _attach_overlay(ds, attach_path)

    monkeypatch.setattr(cli_module, "_is_interactive", lambda: True)
    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "materialize", "demo", "--yes"])

    assert result.exit_code == 0
    assert overlay_path(ds.dir, "audit").exists()


def test_materialize_non_interactive_proceeds(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    _attach_overlay(ds, attach_path)

    monkeypatch.setattr(cli_module, "_is_interactive", lambda: False)
    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "materialize", "demo"])

    assert result.exit_code == 0
    assert overlay_path(ds.dir, "audit").exists()


def test_materialize_drop_overlays(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    _attach_overlay(ds, attach_path)

    monkeypatch.setattr(cli_module, "_is_interactive", lambda: True)
    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "materialize", "demo", "--yes", "--drop-overlays"])

    assert result.exit_code == 0
    assert not overlay_path(ds.dir, "audit").exists()


def test_materialize_snapshot_before(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    _attach_overlay(ds, attach_path)

    snap_dir = ds.snapshot_dir
    snap_dir.mkdir(parents=True, exist_ok=True)
    before = len(list(snap_dir.glob("records-*.parquet")))

    monkeypatch.setattr(cli_module, "_is_interactive", lambda: True)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--root", str(root), "materialize", "demo", "--yes", "--snapshot-before"],
    )

    assert result.exit_code == 0
    after = len(list(snap_dir.glob("records-*.parquet")))
    assert after == before + 2
