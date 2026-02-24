"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_export_cli.py

CLI coverage for `cruncher export sequences`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import dnadesign.cruncher.cli.commands.export as export_module
from dnadesign.cruncher.app.export_sequences_service import SequenceExportResult
from dnadesign.cruncher.cli.app import app

runner = CliRunner()
CONFIG_PATH = Path(__file__).resolve().parents[2] / "workspaces" / "demo_pairwise" / "configs" / "config.yaml"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")


def invoke_cli(args: list[str]):
    return runner.invoke(app, args, color=False)


def combined_output(result) -> str:
    stderr = getattr(result, "stderr", "")
    return ANSI_RE.sub("", f"{result.output}{stderr}")


def test_export_sequences_rejects_run_and_latest() -> None:
    result = invoke_cli(
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
    assert "Use either --run or --latest, not both." in combined_output(result)


def test_export_sequences_passes_cli_options_to_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_export_sequences(
        cfg,
        config_path,
        *,
        runs_override=None,
        use_latest=False,
        table_format="parquet",
    ):
        captured["runs_override"] = runs_override
        captured["use_latest"] = use_latest
        captured["table_format"] = table_format
        out_dir = tmp_path / "run" / "export"
        out_dir.mkdir(parents=True, exist_ok=True)
        return [
            SequenceExportResult(
                run_name="sample_export",
                run_dir=tmp_path / "run",
                output_dir=out_dir,
                manifest_path=out_dir / "export_manifest.json",
                files={
                    "consensus_sites": out_dir / "table__consensus_sites.csv",
                    "elites": out_dir / "table__elites.csv",
                },
                row_counts={
                    "consensus_sites": 2,
                    "elites": 1,
                },
            )
        ]

    monkeypatch.setattr(export_module, "run_export_sequences", _fake_run_export_sequences)

    result = invoke_cli(
        [
            "export",
            "sequences",
            "--latest",
            "--table-format",
            "csv",
            str(CONFIG_PATH),
        ],
    )

    assert result.exit_code == 0
    assert captured.get("runs_override") is None
    assert captured.get("use_latest") is True
    assert captured.get("table_format") == "csv"
    assert "consensus_sites" in result.output
    assert "elites" in result.output


def test_export_sequences_defaults_table_format_to_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_export_sequences(
        cfg,
        config_path,
        *,
        runs_override=None,
        use_latest=False,
        table_format="parquet",
    ):
        captured["table_format"] = table_format
        out_dir = tmp_path / "run" / "export"
        out_dir.mkdir(parents=True, exist_ok=True)
        return [
            SequenceExportResult(
                run_name="sample_export",
                run_dir=tmp_path / "run",
                output_dir=out_dir,
                manifest_path=out_dir / "export_manifest.json",
                files={},
                row_counts={},
            )
        ]

    monkeypatch.setattr(export_module, "run_export_sequences", _fake_run_export_sequences)
    result = invoke_cli(["export", "sequences", "--latest", str(CONFIG_PATH)])

    assert result.exit_code == 0
    assert captured.get("table_format") == "csv"
