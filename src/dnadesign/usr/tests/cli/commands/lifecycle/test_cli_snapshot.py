"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/commands/lifecycle/test_cli_snapshot.py

CLI tests for usr snapshot lifecycle behavior.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.testsupport.usr import ensure_registry
from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset


def _make_dataset(root: Path) -> Dataset:
    ensure_registry(root)
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    return ds


def test_cli_snapshot_creates_snapshot_and_records_event(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)
    before = {path.name for path in ds.snapshot_dir.glob("records-*.parquet")}

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "snapshot", "demo"])

    assert result.exit_code == 0, result.output
    assert "Snapshot saved under" in result.output
    after = {path.name for path in ds.snapshot_dir.glob("records-*.parquet")}
    assert after - before

    events = [json.loads(line) for line in ds.events_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["action"] == "snapshot"


def test_cli_init_requires_registry(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "init", "demo"])

    assert result.exit_code != 0
    assert result.exception is not None
    assert "Registry required for init" in str(result.exception)
