"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/commands/query/test_cli_get.py

CLI tests for usr get record-addressing forms.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.testsupport.usr import ensure_registry
from dnadesign.usr.src.cli import app
from dnadesign.usr.src.dataset import Dataset


def _make_dataset(root: Path) -> Dataset:
    ensure_registry(root)
    ds = Dataset(root, "demo")
    ds.init(source="test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    return ds


def test_cli_get_accepts_positional_record_id(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)
    record_id = str(ds.head(1)["id"].iloc[0])

    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "--root", str(root), "get", "demo", record_id])

    assert result.exit_code == 0, result.output
    assert record_id in result.stdout
    assert "ACGT" in result.stdout


def test_cli_get_default_rich_output_does_not_use_removed_applymap(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)
    record_id = str(ds.head(1)["id"].iloc[0])

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "get", "demo", record_id])

    assert result.exit_code == 0, result.output
    assert record_id in result.stdout
    assert "ACGT" in result.stdout


def test_cli_get_single_record_id_uses_inferred_dataset(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)
    record_id = str(ds.head(1)["id"].iloc[0])
    monkeypatch.chdir(root)

    runner = CliRunner()
    result = runner.invoke(app, ["--no-rich", "--root", str(root), "get", record_id])

    assert result.exit_code == 0, result.output
    assert record_id in result.stdout
    assert "ACGT" in result.stdout


def test_cli_grep_default_rich_output_does_not_use_removed_applymap(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "grep", "demo", "--pattern", "AC"])

    assert result.exit_code == 0, result.output
    assert "ACGT" in result.stdout
