"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_export.py

CLI export command behavior tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.tests.registry_helpers import ensure_registry


def _make_dataset(root: Path) -> None:
    ensure_registry(root)
    ds = Dataset(root, "densegen/demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )


def test_cli_export_parquet_file_target(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)
    out = tmp_path / "demo_export.parquet"

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--root", str(root), "export", "densegen/demo", "--fmt", "parquet", "--out", str(out)],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    table = pq.read_table(out)
    assert table.num_rows == 2


def test_cli_export_directory_target_uses_dataset_filename(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)
    out_dir = tmp_path / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--root", str(root), "export", "densegen/demo", "--fmt", "parquet", "--out", str(out_dir)],
    )

    expected = out_dir / "densegen_demo.parquet"
    assert result.exit_code == 0, result.output
    assert expected.exists()
    assert str(expected) in result.output


def test_cli_export_accepts_absolute_dataset_directory(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)
    dataset_dir = root / "densegen" / "demo"
    out_dir = tmp_path / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--root", str(root), "export", str(dataset_dir), "--fmt", "parquet", "--out", str(out_dir)],
    )

    expected = out_dir / "densegen_demo.parquet"
    assert result.exit_code == 0, result.output
    assert expected.exists()
    assert str(expected) in result.output
