"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_cli_help.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

import dnadesign.cruncher.cli.commands.sources as sources_module
from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex

runner = CliRunner()
CONFIG_PATH = Path(__file__).resolve().parents[1] / "workspaces" / "demo" / "config.yaml"


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
    assert "No config argument provided" in result.output


def test_sources_info_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["sources", "info"])
    assert result.exit_code != 0
    assert "Missing SOURCE" in result.output


def test_sources_datasets_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["sources", "datasets"])
    assert result.exit_code != 0
    assert "Missing SOURCE" in result.output


def test_sources_list_auto_detects_config_in_cwd(tmp_path, monkeypatch) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": ".cruncher"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "ingest": {
                "local_sources": [
                    {
                        "source_id": "demo_local",
                        "root": "local_motifs",
                        "format_map": {".txt": "MEME"},
                    }
                ]
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["sources", "list"])
    assert result.exit_code == 0
    assert "demo_local" in result.output


def test_config_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_config_defaults_to_summary() -> None:
    result = runner.invoke(app, ["config", str(CONFIG_PATH)])
    assert result.exit_code == 0
    assert "Cruncher config summary" in result.output


def test_sources_summary_remote_error_is_user_friendly(tmp_path, monkeypatch) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": ".cruncher"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    class StubRegistry:
        def list_sources(self):
            return [SimpleNamespace(source_id="stub", description="stub")]

        def create(self, source_id, ingest_cfg):
            return object()

    monkeypatch.setattr(sources_module, "default_registry", lambda *args, **kwargs: StubRegistry())

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sources_module, "summarize_remote", _boom)

    result = runner.invoke(
        app,
        ["sources", "summary", "--scope", "remote", str(config_path)],
    )
    assert result.exit_code != 0
    assert "Error: boom" in result.output


def test_sources_summary_cache_filters_source_and_titles(tmp_path) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": ".cruncher"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    index = CatalogIndex()
    index.entries["regulondb:R1"] = CatalogEntry(
        source="regulondb",
        motif_id="R1",
        tf_name="LexA",
        kind="PFM",
        has_matrix=True,
        has_sites=True,
        site_count=5,
        site_total=7,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    index.entries["jaspar:J1"] = CatalogEntry(
        source="jaspar",
        motif_id="J1",
        tf_name="SoxR",
        kind="PFM",
        has_matrix=True,
        has_sites=False,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    index.save(tmp_path / ".cruncher")

    result = runner.invoke(
        app,
        ["sources", "summary", "--scope", "cache", "--source", "regulondb", str(config_path)],
    )
    assert result.exit_code == 0
    assert "Cache overview" in result.output
    assert "Cache regulators" in result.output
    assert "source=regulondb" in result.output
    assert "LexA" in result.output
    assert "SoxR" not in result.output


def test_sources_summary_requires_remote_limit_when_no_iter(tmp_path, monkeypatch) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": ".cruncher"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    class StubAdapter:
        source_id = "stub"

        def capabilities(self):
            return {"motifs:list"}

    class StubRegistry:
        def list_sources(self):
            return [SimpleNamespace(source_id="stub", description="stub")]

        def create(self, source_id, ingest_cfg):
            return StubAdapter()

    monkeypatch.setattr(sources_module, "default_registry", lambda *args, **kwargs: StubRegistry())

    result = runner.invoke(
        app,
        ["sources", "summary", "--scope", "remote", str(config_path)],
    )
    assert result.exit_code != 0
    assert "--remote-limit" in result.output
    assert "stub" in result.output


def test_lock_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["lock"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_parse_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["parse"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_sample_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["sample"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_analyze_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_catalog_show_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["catalog", "show"])
    assert result.exit_code != 0
    assert "Missing REF" in result.output


def test_report_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code != 0
    assert "Missing RUN" in result.output


def test_runs_show_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["runs", "show"])
    assert result.exit_code != 0
    assert "Missing RUN" in result.output


def test_runs_latest_requires_config_with_hint() -> None:
    result = runner.invoke(app, ["runs", "latest"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_runs_watch_requires_args_with_hint() -> None:
    result = runner.invoke(app, ["runs", "watch"])
    assert result.exit_code != 0
    assert "Missing RUN" in result.output


def test_analyze_list_plots_succeeds() -> None:
    result = runner.invoke(app, ["analyze", "--list-plots", str(CONFIG_PATH)])
    assert result.exit_code == 0
    assert "Analysis plot plan" in result.output


def test_analyze_list_plots_without_config_succeeds() -> None:
    result = runner.invoke(app, ["analyze", "--list-plots"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output
