"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_targets_cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _write_config(tmp_path: Path) -> Path:
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["LexA"]],
            "regulator_categories": {
                "CatA": ["LexA", "CpxR"],
                "CatB": ["Fur", "Lrp"],
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "within_category": {"sizes": [2]},
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_targets_list_category(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["targets", "list", "--category", "CatA", str(config_path)], color=False)
    assert result.exit_code == 0
    assert "LexA" in result.output
    assert "CpxR" in result.output


def test_targets_list_campaign(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["targets", "list", "--campaign", "demo", str(config_path)], color=False)
    assert result.exit_code == 0
    assert "LexA" in result.output
    assert "Fur" in result.output


def test_targets_list_rejects_category_and_campaign(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["targets", "list", "--category", "CatA", "--campaign", "demo", str(config_path)],
        color=False,
    )
    assert result.exit_code != 0
    assert "Use either --category or --campaign" in result.output
