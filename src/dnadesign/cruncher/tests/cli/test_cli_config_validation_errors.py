"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_cli_config_validation_errors.py

CLI contract tests for config-schema validation failures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _write_invalid_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "bad.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "cruncher:",
                "  schema_version: 3",
                "  workspace:",
                "    out_dir: outputs",
                "    regulator_sets: [[lexA, cpxR]]",
                "  sample:",
                "    seed: 7",
                "  unknown_section: {}",
                "",
            ]
        )
    )
    return config_path


@pytest.mark.parametrize(
    "argv",
    [
        ["config", "summary", "--config"],
        ["parse", "--config"],
        ["sample", "--config"],
        ["analyze", "--config"],
        ["lock", "--config"],
        ["status", "--config"],
        ["export", "sequences", "--config"],
        ["fetch", "motifs", "--config", "--tf", "lexA"],
        ["fetch", "sites", "--config"],
        ["discover", "motifs", "--config", "--tf", "lexA"],
        ["discover", "check", "--config"],
        ["doctor", "--config"],
        ["runs", "list", "--config"],
        ["cache", "stats", "--config"],
        ["catalog", "list", "--config"],
        ["sources", "list", "--config"],
        ["sources", "info", "regulondb", "--config"],
        ["sources", "datasets", "regulondb", "--config"],
        ["sources", "summary", "--config"],
        ["targets", "list", "--config"],
        ["targets", "status", "--config"],
    ],
)
def test_primary_cli_commands_report_clean_config_validation_errors(tmp_path: Path, argv: list[str]) -> None:
    bad_config = _write_invalid_config(tmp_path)
    args = list(argv)
    config_idx = args.index("--config")
    args.insert(config_idx + 1, str(bad_config))
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output
