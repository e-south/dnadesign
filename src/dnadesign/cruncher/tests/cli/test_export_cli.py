"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_export_cli.py

CLI coverage for `cruncher export sequences`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import dnadesign.cruncher.cli.commands.export as export_module
from dnadesign.cruncher.app.export_sequences_service import SequenceExportResult
from dnadesign.cruncher.cli.app import app

runner = CliRunner()
CONFIG_PATH = Path(__file__).resolve().parents[2] / "workspaces" / "demo_basics_two_tf" / "config.yaml"


def test_export_sequences_rejects_run_and_latest() -> None:
    result = runner.invoke(
        app,
        [
            "export",
            "sequences",
            "--run",
            "sample_run",
            "--latest",
            str(CONFIG_PATH),
        ],
    )
    assert result.exit_code != 0
    assert "Use either --run or --latest, not both." in result.output


def test_export_sequences_rejects_small_combo_size() -> None:
    result = runner.invoke(
        app,
        [
            "export",
            "sequences",
            "--latest",
            "--max-combo-size",
            "1",
            str(CONFIG_PATH),
        ],
    )
    assert result.exit_code != 0
    assert "--max-combo-size must be >= 2." in result.output


def test_export_sequences_passes_cli_options_to_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_export_sequences(
        cfg,
        config_path,
        *,
        runs_override=None,
        use_latest=False,
        table_format="parquet",
        max_combo_size=None,
    ):
        captured["runs_override"] = runs_override
        captured["use_latest"] = use_latest
        captured["table_format"] = table_format
        captured["max_combo_size"] = max_combo_size
        out_dir = tmp_path / "run" / "export" / "sequences"
        out_dir.mkdir(parents=True, exist_ok=True)
        return [
            SequenceExportResult(
                run_name="sample_export",
                run_dir=tmp_path / "run",
                output_dir=out_dir,
                manifest_path=out_dir / "export_manifest.json",
                files={
                    "consensus_sites": out_dir / "table__consensus_sites.csv",
                    "elite_tf_windows": out_dir / "table__elite_tf_windows.csv",
                    "elite_pairwise_windows": out_dir / "table__elite_pairwise_windows.csv",
                    "elite_combinations": out_dir / "table__elite_combinations.csv",
                },
                row_counts={
                    "consensus_sites": 2,
                    "elite_tf_windows": 2,
                    "elite_pairwise_windows": 1,
                    "elite_combinations": 1,
                },
            )
        ]

    monkeypatch.setattr(export_module, "run_export_sequences", _fake_run_export_sequences)

    result = runner.invoke(
        app,
        [
            "export",
            "sequences",
            "--latest",
            "--table-format",
            "csv",
            "--max-combo-size",
            "3",
            str(CONFIG_PATH),
        ],
    )

    assert result.exit_code == 0
    assert captured.get("runs_override") is None
    assert captured.get("use_latest") is True
    assert captured.get("table_format") == "csv"
    assert captured.get("max_combo_size") == 3
    assert "consensus_sites" in result.output
