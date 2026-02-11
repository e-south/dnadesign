"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_legacy_paths.py

CLI tests enforcing hard errors for legacy USR dataset paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.usr.src.cli import app


def _write_legacy_dataset(root: Path) -> Path:
    legacy_dir = root / "archived" / "legacy_demo"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": [0],
            "sequence": ["ACGT"],
            "bio_type": ["dna"],
            "alphabet": ["dna_4"],
            "source": ["test"],
        }
    )
    pq.write_table(table, legacy_dir / "records.parquet")
    (legacy_dir / "meta.md").write_text("name: archived/legacy_demo\n", encoding="utf-8")
    return legacy_dir


def test_cli_info_rejects_legacy_dataset_path_in_archived_root(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    legacy_dir = _write_legacy_dataset(root)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "info",
            str(legacy_dir),
            "--format",
            "json",
        ],
    )
    assert result.exit_code != 0
    detail = result.stdout or str(result.exception)
    assert "Legacy dataset paths under 'archived/' are not supported" in detail


def test_cli_head_rejects_legacy_dataset_path_in_archived_root(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    legacy_dir = _write_legacy_dataset(root)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "head",
            str(legacy_dir),
            "-n",
            "1",
        ],
    )
    assert result.exit_code != 0
    detail = result.stdout or str(result.exception)
    assert "Legacy dataset paths under 'archived/' are not supported" in detail
