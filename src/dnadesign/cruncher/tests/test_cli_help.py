from pathlib import Path

from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_root_help_includes_command_descriptions() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fetch" in result.output
    assert "Fetch motifs or binding sites" in result.output


def test_fetch_motifs_requires_tf_or_motif_id() -> None:
    result = runner.invoke(app, ["fetch", "motifs", str(CONFIG_PATH)])
    assert result.exit_code != 0
    assert "Provide at least one --tf or --motif-id" in result.output


def test_catalog_show_requires_source_ref() -> None:
    result = runner.invoke(app, ["catalog", "show", str(CONFIG_PATH), "badref"])
    assert result.exit_code != 0
    assert "Expected <source>:<motif_id>" in result.output
