"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_cli_help.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import dnadesign.cruncher.app.analyze_workflow as analyze_workflow
import dnadesign.cruncher.cli.commands.sources as sources_module
from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.cli.config_resolver import (
    CONFIG_ENV_VAR,
    DEFAULT_WORKSPACE_ENV_VAR,
    NONINTERACTIVE_ENV_VAR,
    WORKSPACE_ENV_VAR,
    WORKSPACE_ROOTS_ENV_VAR,
)
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
runner = CliRunner()
CONFIG_PATH = Path(__file__).resolve().parents[1] / "workspaces" / "demo_basics_two_tf" / "config.yaml"


def invoke_cli(args: list[str], env: dict[str, str] | None = None):
    return runner.invoke(app, args, env=env, color=False)


def combined_output(result) -> str:
    stderr = getattr(result, "stderr", "")
    return ANSI_RE.sub("", f"{result.output}{stderr}")


def invoke_isolated(args: list[str], env: dict[str, str] | None = None):
    merged_env = {NONINTERACTIVE_ENV_VAR: "1"}
    if env:
        merged_env.update(env)
    with runner.isolated_filesystem():
        return invoke_cli(args, env=merged_env)


@pytest.fixture(autouse=True)
def _clear_workspace_env(monkeypatch: pytest.MonkeyPatch):
    for var in (
        CONFIG_ENV_VAR,
        WORKSPACE_ENV_VAR,
        DEFAULT_WORKSPACE_ENV_VAR,
        WORKSPACE_ROOTS_ENV_VAR,
        NONINTERACTIVE_ENV_VAR,
    ):
        monkeypatch.delenv(var, raising=False)


def test_root_help_includes_command_descriptions() -> None:
    result = invoke_cli(["--help"])
    assert result.exit_code == 0
    assert "fetch" in result.output
    assert "Fetch motifs/sites" in result.output
    assert "status" in result.output
    assert "workspaces" in result.output


def test_workspaces_list_includes_demo() -> None:
    result = invoke_cli(["workspaces", "list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "demo_basics_two_tf" in result.output


def test_fetch_motifs_requires_tf_or_motif_id() -> None:
    result = invoke_cli(["fetch", "motifs", str(CONFIG_PATH)])
    assert result.exit_code != 0
    assert "Provide at least one --tf, --motif-id, or --campaign" in combined_output(result)


def test_fetch_motifs_rejects_campaign_and_tf() -> None:
    result = invoke_cli(["fetch", "motifs", "--campaign", "demo_pair", "--tf", "lexA", str(CONFIG_PATH)])
    assert result.exit_code != 0
    assert "--campaign cannot be combined with --tf or --motif-id" in combined_output(result)


def test_global_config_option_resolves_workspace() -> None:
    result = invoke_cli(["-c", str(CONFIG_PATH), "sources", "list"])
    assert result.exit_code == 0
    assert "demo_local_meme" in result.output


def test_targets_status_rejects_site_kinds_with_matrix_pwm(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = invoke_cli(["targets", "status", "--site-kind", "curated", str(config_path)])
    assert result.exit_code != 0
    assert "--site-kind requires pwm_source=sites" in combined_output(result)


def test_campaign_generate_resolves_relative_out_to_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA"]],
            "regulator_categories": {"A": ["lexA"], "B": ["cpxR"]},
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["A", "B"],
                    "across_categories": {"sizes": [2]},
                }
            ],
            "motif_store": {"catalog_root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    result = invoke_cli(["campaign", "generate", "--campaign", "demo", "--out", "derived.yaml", str(config_path)])
    assert result.exit_code == 0
    assert (tmp_path / "derived.yaml").exists()


def test_campaign_generate_rejects_outside_workspace(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA"]],
            "regulator_categories": {"A": ["lexA"], "B": ["cpxR"]},
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["A", "B"],
                    "across_categories": {"sizes": [2]},
                }
            ],
            "motif_store": {"catalog_root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    outside = tmp_path / "outside"
    outside.mkdir()
    out_path = outside / "derived.yaml"

    result = invoke_cli(["campaign", "generate", "--campaign", "demo", "--out", str(out_path), str(config_path)])
    assert result.exit_code != 0
    assert "--out must be inside the workspace" in combined_output(result)


def test_catalog_show_requires_source_ref() -> None:
    result = invoke_cli(["catalog", "show", str(CONFIG_PATH), "badref"])
    assert result.exit_code != 0
    assert "Expected <source>:<motif_id>" in combined_output(result)


def test_status_requires_config_with_hint() -> None:
    result = invoke_isolated(["status"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_sources_info_requires_args_with_hint() -> None:
    result = invoke_cli(["sources", "info"])
    assert result.exit_code != 0
    assert "Missing SOURCE" in result.output


def test_sources_datasets_requires_args_with_hint() -> None:
    result = invoke_cli(["sources", "datasets"])
    assert result.exit_code != 0
    assert "Missing SOURCE" in result.output


def test_sources_list_auto_detects_config_in_cwd(tmp_path, monkeypatch) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": str(tmp_path / ".cruncher")},
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

    result = invoke_cli(["sources", "list"])
    assert result.exit_code == 0
    assert "demo_local" in result.output


def test_config_requires_config_with_hint() -> None:
    result = invoke_isolated(["config"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_config_defaults_to_summary() -> None:
    result = invoke_cli(["config", str(CONFIG_PATH)])
    assert result.exit_code == 0
    assert "Cruncher config summary" in result.output


def test_analyze_hint_not_duplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise ValueError("No analysis runs configured. Set analysis.runs, pass --run, or use --latest.")

    monkeypatch.setattr(analyze_workflow, "run_analyze", _boom)

    result = invoke_cli(["analyze", str(CONFIG_PATH)])
    assert result.exit_code != 0
    assert "No analysis runs configured" in result.output
    assert "Hint: set analysis.runs" not in result.output


def test_sources_summary_remote_error_is_user_friendly(tmp_path, monkeypatch) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": str(tmp_path / ".cruncher")},
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

    result = invoke_cli(["sources", "summary", "--scope", "remote", str(config_path)])
    assert result.exit_code != 0
    assert "Error: boom" in result.output


def test_sources_summary_cache_filters_source_and_titles(tmp_path) -> None:
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": str(tmp_path / ".cruncher")},
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

    result = invoke_cli(["sources", "summary", "--scope", "cache", "--source", "regulondb", str(config_path)])
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
            "motif_store": {"catalog_root": str(tmp_path / ".cruncher")},
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

    result = invoke_cli(["sources", "summary", "--scope", "remote", str(config_path)])
    assert result.exit_code != 0
    assert "--remote-limit" in result.output
    assert "stub" in result.output


def test_lock_requires_config_with_hint() -> None:
    result = invoke_isolated(["lock"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_parse_requires_config_with_hint() -> None:
    result = invoke_isolated(["parse"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_sample_requires_config_with_hint() -> None:
    result = invoke_isolated(["sample"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_analyze_requires_config_with_hint() -> None:
    result = invoke_isolated(["analyze"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_catalog_show_requires_args_with_hint() -> None:
    result = invoke_cli(["catalog", "show"])
    assert result.exit_code != 0
    assert "Missing REF" in result.output


def test_report_requires_args_with_hint() -> None:
    result = invoke_cli(["report"])
    assert result.exit_code != 0
    assert "Missing RUN" in result.output


def test_runs_show_requires_args_with_hint() -> None:
    result = invoke_cli(["runs", "show"])
    assert result.exit_code != 0
    assert "Missing RUN" in result.output


def test_runs_latest_requires_config_with_hint() -> None:
    result = invoke_isolated(["runs", "latest"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output


def test_runs_watch_requires_args_with_hint() -> None:
    result = invoke_cli(["runs", "watch"])
    assert result.exit_code != 0
    assert "Missing RUN" in result.output


def test_analyze_list_plots_succeeds() -> None:
    result = invoke_cli(["analyze", "--list-plots", str(CONFIG_PATH)])
    assert result.exit_code == 0
    assert "Analysis plot plan" in result.output


def test_analyze_list_plots_without_config_succeeds() -> None:
    result = invoke_isolated(["analyze", "--list-plots"])
    assert result.exit_code != 0
    assert "No config argument provided" in result.output
