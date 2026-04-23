"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_cli_state.py

CLI tests for usr state commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.usr import Dataset
from dnadesign.usr.src.cli import app
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def _make_dataset(root: Path) -> Dataset:
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


def test_cli_state_set_and_clear(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = _make_dataset(root)
    record_id = ds.head(1)["id"].iloc[0]

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "state",
            "set",
            "demo",
            "--id",
            record_id,
            "--masked",
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "state",
            "clear",
            "demo",
            "--id",
            record_id,
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "state",
            "get",
            "demo",
            "--id",
            record_id,
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
