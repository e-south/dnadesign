"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_cli_help.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import dnadesign.cruncher.app.analyze_workflow as analyze_workflow
import dnadesign.cruncher.cli.commands.sources as sources_module
from dnadesign.cruncher.artifacts.layout import config_used_path, elites_path, sequences_path
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
CONFIG_PATH = Path(__file__).resolve().parents[2] / "workspaces" / "demo_basics_two_tf" / "config.yaml"


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
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = invoke_cli(["targets", "status", "--site-kind", "curated", str(config_path)])
    assert result.exit_code != 0
    assert "--site-kind requires pwm_source=sites" in combined_output(result)


def test_campaign_generate_defaults_to_workspace_campaign_state(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [],
                "regulator_categories": {"A": ["lexA"], "B": ["cpxR"]},
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["A", "B"],
                    "across_categories": {"sizes": [2]},
                }
            ],
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = invoke_cli(["campaign", "generate", "--campaign", "demo", str(config_path)])
    assert result.exit_code == 0
    generated = tmp_path / ".cruncher" / "campaigns" / "demo" / "generated.yaml"
    manifest = tmp_path / ".cruncher" / "campaigns" / "demo" / "generated.campaign_manifest.json"
    assert generated.exists()
    assert manifest.exists()


def test_campaign_generate_rejects_outside_workspace(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [],
                "regulator_categories": {"A": ["lexA"], "B": ["cpxR"]},
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["A", "B"],
                    "across_categories": {"sizes": [2]},
                }
            ],
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    out_path = outside / "derived.yaml"

    result = invoke_cli(["campaign", "generate", "--campaign", "demo", "--out", str(out_path), str(config_path)])
    assert result.exit_code != 0
    assert "--out must be inside" in combined_output(result)


def test_lock_campaign_required_for_campaign_mode_config(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [],
                "regulator_categories": {"A": ["lexA"], "B": ["cpxR"]},
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["A", "B"],
                    "across_categories": {"sizes": [2]},
                }
            ],
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = invoke_cli(["lock", str(config_path)])
    assert result.exit_code != 0
    assert "--campaign" in combined_output(result)


def test_sample_campaign_required_for_campaign_mode_config(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [],
                "regulator_categories": {"A": ["lexA"], "B": ["cpxR"]},
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["A", "B"],
                    "across_categories": {"sizes": [2]},
                }
            ],
            "sample": {
                "seed": 1,
                "sequence_length": 12,
                "budget": {"tune": 1, "draws": 1},
                "pt": {"n_temps": 2, "temp_max": 5.0},
                "elites": {"k": 1},
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = invoke_cli(["sample", str(config_path)])
    assert result.exit_code != 0
    assert "--campaign" in combined_output(result)


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
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher")},
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


def test_config_reports_invalid_schema_without_traceback(tmp_path: Path) -> None:
    bad_config = {
        "cruncher": {
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher")},
        }
    }
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(yaml.safe_dump(bad_config))

    result = invoke_cli(["config", str(config_path)])

    assert result.exit_code != 0
    assert not isinstance(result.exception, ValueError)
    assert "Error: Config schema v3 required (schema_version: 3)" in result.output


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
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher")},
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
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher")},
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
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher")},
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


def test_analyze_latest_requires_complete_artifacts(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA", "cpxR"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
            "sample": {
                "seed": 7,
                "sequence_length": 12,
                "budget": {"tune": 1, "draws": 1},
                "elites": {"k": 1},
                "output": {"save_sequences": True, "save_trace": False},
            },
            "analysis": {
                "run_selector": "latest",
                "pairwise": "off",
                "plot_format": "png",
                "plot_dpi": 72,
                "table_format": "parquet",
                "max_points": 1000,
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = tmp_path / "results" / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_used = {
        "cruncher": {
            "schema_version": 3,
            "active_regulator_set": {"tfs": ["lexA", "cpxR"]},
            "pwms_info": {
                "lexA": {"pwm_matrix": [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]},
                "cpxR": {"pwm_matrix": [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]},
            },
            "sample": {"objective": {"score_scale": "normalized-llr"}},
        }
    }
    config_used_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    config_used_path(run_dir).write_text(yaml.safe_dump(config_used))

    sequences_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "chain": [0],
            "draw": [0],
            "phase": ["draw"],
            "sequence": ["ACGTACGTACGT"],
            "score_lexA": [1.0],
            "score_cpxR": [1.1],
        }
    ).to_parquet(sequences_path(run_dir), engine="fastparquet")
    elites_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ["elite-1"],
            "sequence": ["ACGTACGTACGT"],
            "rank": [1],
            "score_lexA": [1.0],
            "score_cpxR": [1.1],
        }
    ).to_parquet(elites_path(run_dir), engine="fastparquet")

    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "stage": "sample",
                "run_dir": str(run_dir.resolve()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "draws": 1,
                "adapt_sweeps": 1,
                "top_k": 1,
                "objective": {"bidirectional": True, "score_scale": "normalized-llr"},
                "optimizer_stats": {"beta_ladder_final": [1.0]},
                "artifacts": [],
            },
            indent=2,
        )
    )

    result = invoke_cli(["analyze", "--latest", str(config_path)])
    assert result.exit_code != 0
    assert "Missing elites hits parquet" in result.output


def test_analyze_rejects_run_and_latest_together() -> None:
    result = invoke_cli(["analyze", "--latest", "--run", "sample_123", str(CONFIG_PATH)])
    assert result.exit_code != 0
    assert "Use either --run or --latest, not both." in result.output


def test_analyze_accepts_run_directory_path_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    run_dir = tmp_path / "runs" / "sample" / "sample_path"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fake_run_analyze(cfg, config_path, *, runs_override=None, use_latest=False):
        captured["runs_override"] = runs_override
        captured["use_latest"] = use_latest
        return []

    monkeypatch.setattr(analyze_workflow, "run_analyze", _fake_run_analyze)

    result = invoke_cli(["analyze", "--run", str(run_dir.resolve()), str(CONFIG_PATH)])

    assert result.exit_code == 0
    assert captured.get("runs_override") == [str(run_dir.resolve())]
    assert captured.get("use_latest") is False
