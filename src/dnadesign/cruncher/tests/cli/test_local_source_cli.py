"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_local_source_cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def test_sources_list_includes_local_sources(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": ".cruncher"},
            "ingest": {
                "local_sources": [
                    {
                        "source_id": "local_omalle",
                        "root": "motifs",
                        "patterns": ["*.txt"],
                        "format_map": {".txt": "MEME"},
                    }
                ]
            },
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = runner.invoke(app, ["sources", "list", str(config_path)])
    assert result.exit_code == 0
    assert "local_omalle" in result.output
