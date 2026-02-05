"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_cli_typer.py

Typer CLI integration tests for USR.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset


def _make_dataset(root: Path) -> None:
    ds = Dataset(root, "demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )


def test_cols_accepts_dataset_name(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)
    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "cols", "demo"])
    assert result.exit_code == 0
    assert "sequence" in result.stdout
