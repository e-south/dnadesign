"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_sources_cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def test_sources_list_uses_workspace_config(tmp_path: Path) -> None:
    roots = tmp_path / "roots"
    workspace = roots / "demo_test"
    workspace.mkdir(parents=True)

    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "ingest": {
                "local_sources": [
                    {
                        "source_id": "local_demo",
                        "description": "Local demo motifs",
                        "root": "data/local",
                        "format_map": {".txt": "MEME"},
                    }
                ]
            },
        }
    }
    config_path = workspace / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config))

    result = runner.invoke(
        app,
        ["--workspace", "demo_test", "sources", "list"],
        env={"CRUNCHER_WORKSPACE_ROOTS": str(roots)},
    )
    assert result.exit_code == 0
    assert "local_demo" in result.output
