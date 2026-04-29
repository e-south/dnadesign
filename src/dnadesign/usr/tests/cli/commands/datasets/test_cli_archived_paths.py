"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/commands/datasets/test_cli_archived_paths.py

CLI tests for the sanctioned archived dataset storage root.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.devtools.tests.support.usr import ensure_registry
from dnadesign.usr.src.cli import app


def _write_archived_dataset(root: Path) -> Path:
    archive_root = root / "archived"
    ensure_registry(archive_root)
    archived_dir = archive_root / "legacy_demo"
    archived_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": [0],
            "sequence": ["ACGT"],
            "bio_type": ["dna"],
            "alphabet": ["dna_4"],
            "source": ["test"],
        }
    )
    pq.write_table(table, archived_dir / "records.parquet")
    (archived_dir / "meta.md").write_text("name: legacy_demo\n", encoding="utf-8")
    return archived_dir


def test_cli_info_accepts_explicit_archived_dataset_path(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    archived_dir = _write_archived_dataset(root)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "info",
            str(archived_dir),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert '"name":"legacy_demo"' in result.stdout
    assert '"path":"' in result.stdout


def test_cli_head_accepts_explicit_archived_dataset_path(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    archived_dir = _write_archived_dataset(root)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "head",
            str(archived_dir),
            "-n",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "ACGT" in result.stdout


def test_cli_info_rejects_archived_dataset_id_prefix(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _write_archived_dataset(root)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "info",
            "archived/legacy_demo",
            "--format",
            "json",
        ],
    )
    assert result.exit_code != 0
    detail = result.stdout or str(result.exception)
    assert "reserved for archived storage" in detail
