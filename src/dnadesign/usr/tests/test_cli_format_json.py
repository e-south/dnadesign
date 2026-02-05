"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_cli_format_json.py

CLI JSON output format tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
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
        ],
        source="unit",
    )


def test_cli_info_json_format(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "info", "demo", "--format", "json"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["usr_output_version"] == 1
    assert payload["data"]["name"] == "demo"


def test_cli_ls_json_format(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _make_dataset(root)

    runner = CliRunner()
    result = runner.invoke(app, ["--root", str(root), "ls", "--format", "json"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["usr_output_version"] == 1
    datasets = [row["dataset"] for row in payload["data"]]
    assert "demo" in datasets
