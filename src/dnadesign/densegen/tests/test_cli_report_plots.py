from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app
from dnadesign.densegen.tests.config_fixtures import write_minimal_config


def test_report_requires_plot_manifest_when_enabled(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    write_minimal_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["report", "--plots", "include", "-c", str(cfg_path)])
    assert result.exit_code != 0
    assert "plot_manifest" in result.output
