"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/commands/lifecycle/test_cli_state.py

CLI tests for usr state commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.devtools.tests.support.usr import register_test_namespace
from dnadesign.usr import Dataset
from dnadesign.usr.src.cli import app


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


def test_cli_delete_and_restore_roundtrip_updates_live_view(tmp_path: Path) -> None:
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
            "delete",
            "demo",
            "--id",
            record_id,
            "--reason",
            "adversarial smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    assert record_id not in ds.head(10)["id"].tolist()
    assert record_id in ds.head(10, include_deleted=True)["id"].tolist()

    result = runner.invoke(app, ["--root", str(root), "restore", "demo", "--id", record_id])
    assert result.exit_code == 0, result.output
    assert record_id in ds.head(10)["id"].tolist()


def test_cli_state_get_json_uses_null_for_missing_state_values(tmp_path: Path) -> None:
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
            "--qc-status",
            "pass",
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
    assert "NaN" not in result.stdout

    payload = json.loads(result.stdout)
    row = payload["data"][0]
    assert row["usr_state__qc_status"] == "pass"
    assert row["usr_state__split"] is None
    assert row["usr_state__supersedes"] is None

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
