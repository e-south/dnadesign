from pathlib import Path

from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_root_help_includes_command_descriptions() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fetch" in result.output
    assert "Fetch motifs/sites" in result.output
    assert "status" in result.output


def test_fetch_motifs_requires_tf_or_motif_id() -> None:
    result = runner.invoke(app, ["fetch", "motifs", str(CONFIG_PATH)])
    assert result.exit_code != 0
    assert "Provide at least one --tf or --motif-id" in result.output


def test_catalog_show_requires_source_ref() -> None:
    result = runner.invoke(app, ["catalog", "show", str(CONFIG_PATH), "badref"])
    assert result.exit_code != 0
    assert "Expected <source>:<motif_id>" in result.output


def test_status_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code != 0
    assert "Missing CONFIG" in result.output


def test_sources_info_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["sources", "info"])
    assert result.exit_code != 0
    assert "Missing SOURCE/CONFIG" in result.output


def test_sources_datasets_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["sources", "datasets"])
    assert result.exit_code != 0
    assert "Missing SOURCE/CONFIG" in result.output


def test_lock_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["lock"])
    assert result.exit_code != 0
    assert "Missing CONFIG" in result.output


def test_parse_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["parse"])
    assert result.exit_code != 0
    assert "Missing CONFIG" in result.output


def test_sample_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["sample"])
    assert result.exit_code != 0
    assert "Missing CONFIG" in result.output


def test_analyze_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code != 0
    assert "Missing CONFIG" in result.output


def test_catalog_show_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["catalog", "show"])
    assert result.exit_code != 0
    assert "Missing CONFIG/REF" in result.output


def test_report_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code != 0
    assert "Missing CONFIG/RUN" in result.output


def test_runs_show_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["runs", "show"])
    assert result.exit_code != 0
    assert "Missing CONFIG/RUN" in result.output


def test_runs_latest_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["runs", "latest"])
    assert result.exit_code != 0
    assert "Missing CONFIG" in result.output


def test_runs_watch_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["runs", "watch"])
    assert result.exit_code != 0
    assert "Missing CONFIG/RUN" in result.output


def test_analyze_list_plots_succeeds() -> None:
    result = runner.invoke(app, ["analyze", "--list-plots", str(CONFIG_PATH)])
    assert result.exit_code == 0
    assert "Analysis plot plan" in result.output
