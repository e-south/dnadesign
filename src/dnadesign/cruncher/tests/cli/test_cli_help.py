"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_cli_help.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import dnadesign.cruncher.app.analyze_workflow as analyze_workflow
import dnadesign.cruncher.cli.commands.discover as discover_module
import dnadesign.cruncher.cli.commands.sources as sources_module
from dnadesign.cruncher.artifacts.layout import config_used_path, elites_path, manifest_path, sequences_path
from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.cli.config_resolver import (
    CONFIG_ENV_VAR,
    DEFAULT_WORKSPACE_ENV_VAR,
    NONINTERACTIVE_ENV_VAR,
    WORKSPACE_ENV_VAR,
    WORKSPACE_ROOTS_ENV_VAR,
)
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.study.manifest import (
    StudyManifestV1,
    StudyStatusV1,
    StudyTrialRun,
    write_study_manifest,
    write_study_status,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
runner = CliRunner()
CONFIG_PATH = Path(__file__).resolve().parents[2] / "workspaces" / "demo_pairwise" / "configs" / "config.yaml"


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


def test_export_command_module_defers_service_import() -> None:
    command_module = "dnadesign.cruncher.cli.commands.export"
    service_module = "dnadesign.cruncher.app.export_sequences_service"
    sys.modules.pop(command_module, None)
    sys.modules.pop(service_module, None)

    importlib.import_module(command_module)

    assert service_module not in sys.modules


def test_study_command_module_defers_workflow_import() -> None:
    command_module = "dnadesign.cruncher.cli.commands.study"
    workflow_module = "dnadesign.cruncher.app.study_workflow"
    sys.modules.pop(command_module, None)
    sys.modules.pop(workflow_module, None)

    importlib.import_module(command_module)

    assert workflow_module not in sys.modules


def test_portfolio_command_module_defers_workflow_import() -> None:
    command_module = "dnadesign.cruncher.cli.commands.portfolio"
    workflow_module = "dnadesign.cruncher.app.portfolio_workflow"
    sys.modules.pop(command_module, None)
    sys.modules.pop(workflow_module, None)

    importlib.import_module(command_module)

    assert workflow_module not in sys.modules


def test_analyze_command_module_defers_config_load_import() -> None:
    command_module = "dnadesign.cruncher.cli.commands.analyze"
    config_load_module = "dnadesign.cruncher.config.load"
    sys.modules.pop(command_module, None)
    sys.modules.pop(config_load_module, None)

    importlib.import_module(command_module)

    assert config_load_module not in sys.modules


def test_workspaces_list_includes_demo() -> None:
    result = invoke_cli(["workspaces", "list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "demo_pairwise" in result.output


def test_workspaces_clean_transient_dry_run_preserves_files(tmp_path: Path) -> None:
    root = tmp_path / "workspace_root"
    pycache_dir = root / "pkg" / "__pycache__"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    pyc_file = pycache_dir / "mod.cpython-312.pyc"
    pyc_file.write_bytes(b"pyc")
    ds_store = root / ".DS_Store"
    ds_store.write_text("meta")

    result = invoke_cli(["workspaces", "clean-transient", "--root", str(root)])
    assert result.exit_code == 0
    output = combined_output(result)
    assert "Dry run only" in output
    assert pycache_dir.exists()
    assert pyc_file.exists()
    assert ds_store.exists()


def test_workspaces_clean_transient_confirm_deletes_files(tmp_path: Path) -> None:
    root = tmp_path / "workspace_root"
    pycache_dir = root / "pkg" / "__pycache__"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    pyc_file = pycache_dir / "mod.cpython-312.pyc"
    pyc_file.write_bytes(b"pyc")
    ds_store = root / ".DS_Store"
    ds_store.write_text("meta")

    result = invoke_cli(["workspaces", "clean-transient", "--root", str(root), "--confirm"])
    assert result.exit_code == 0
    output = combined_output(result)
    assert "Deleted" in output
    assert not pycache_dir.exists()
    assert not pyc_file.exists()
    assert not ds_store.exists()


def test_workspaces_clean_transient_include_catalog_cache_deletes_dot_cruncher(tmp_path: Path) -> None:
    root = tmp_path / "workspace_root"
    catalog_dir = root / ".cruncher"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "index.json").write_text("{}")

    result = invoke_cli(
        [
            "workspaces",
            "clean-transient",
            "--root",
            str(root),
            "--include-catalog-cache",
            "--confirm",
        ]
    )
    assert result.exit_code == 0
    output = combined_output(result)
    assert "Deleted" in output
    assert not catalog_dir.exists()


def test_workspaces_list_includes_study_counts(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    workspace = roots / "demo"
    config_dir = workspace / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "cruncher: {schema_version: 3, workspace: {out_dir: outputs, regulator_sets: [[lexA,cpxR]]}}\n"
    )

    spec_path = workspace / "configs" / "studies" / "diag.study.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "study": {
                    "schema_version": 3,
                    "name": "diag",
                    "base_config": "config.yaml",
                    "target": {"kind": "regulator_set", "set_index": 1},
                    "execution": {
                        "parallelism": 1,
                        "on_trial_error": "continue",
                        "exit_code_policy": "nonzero_if_any_error",
                        "summarize_after_run": True,
                    },
                    "artifacts": {"trial_output_profile": "minimal"},
                    "replicates": {"seed_path": "sample.seed", "seeds": [1]},
                    "trials": [{"id": "L6", "factors": {"sample.sequence_length": 6}}],
                }
            }
        )
    )

    run_dir = workspace / "outputs" / "studies" / "diag" / "study123"
    manifest_file = run_dir / "study" / "study_manifest.json"
    status_file = run_dir / "study" / "study_status.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    write_study_manifest(
        manifest_file,
        StudyManifestV1(
            study_name="diag",
            study_id="study123",
            spec_path=str(spec_path),
            spec_sha256="spec",
            base_config_path=str(config_path),
            base_config_sha256="cfg",
            created_at="2026-02-17T00:00:00+00:00",
            trial_runs=[
                StudyTrialRun(
                    trial_id="L6",
                    seed=1,
                    target_set_index=1,
                    target_tfs=["lexA", "cpxR"],
                    status="success",
                    run_dir=str(run_dir / "trials" / "L6" / "seed_1" / "run_a"),
                )
            ],
        ),
    )
    write_study_status(
        status_file,
        StudyStatusV1(
            study_name="diag",
            study_id="study123",
            status="completed",
            total_runs=1,
            pending_runs=0,
            running_runs=0,
            success_runs=1,
            error_runs=0,
            skipped_runs=0,
            warnings=[],
            started_at="2026-02-17T00:00:00+00:00",
            updated_at="2026-02-17T00:00:00+00:00",
            finished_at="2026-02-17T00:00:00+00:00",
        ),
    )

    result = invoke_cli(
        ["workspaces", "list"],
        env={WORKSPACE_ROOTS_ENV_VAR: str(roots), NONINTERACTIVE_ENV_VAR: "1", "COLUMNS": "200"},
    )
    assert result.exit_code == 0
    output = combined_output(result)
    assert "Study Specs" in output
    assert "Study Runs" in output
    assert "demo" in output


def test_fetch_motifs_requires_tf_or_motif_id() -> None:
    result = invoke_cli(["fetch", "motifs", str(CONFIG_PATH)])
    assert result.exit_code != 0
    assert "Provide at least one --tf or --motif-id" in combined_output(result)


def test_fetch_motifs_rejects_campaign_option() -> None:
    result = invoke_cli(["fetch", "motifs", "--campaign", "demo_pair", str(CONFIG_PATH)])
    assert result.exit_code != 0
    assert "No such option: --campaign" in combined_output(result)


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


def test_sample_verbose_uses_runtime_progress_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
            "sample": {
                "seed": 1,
                "sequence_length": 12,
                "budget": {"tune": 1, "draws": 1},
                "optimizer": {"kind": "gibbs_anneal", "chains": 1, "cooling": {"kind": "fixed", "beta": 1.0}},
                "elites": {"k": 1},
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    calls: list[dict[str, object]] = []

    def _fake_run_sample(
        cfg,
        config_path_in: Path,
        *,
        force_overwrite: bool = False,
        progress_bar: bool = True,
        progress_every: int = 0,
    ) -> None:
        calls.append(
            {
                "config_path": config_path_in,
                "force_overwrite": force_overwrite,
                "progress_bar": progress_bar,
                "progress_every": progress_every,
            }
        )

    monkeypatch.setattr("dnadesign.cruncher.app.sample_workflow.run_sample", _fake_run_sample)

    result = invoke_cli(["sample", "--verbose", str(config_path)])
    assert result.exit_code == 0
    assert calls
    assert calls[0]["config_path"] == config_path
    assert calls[0]["force_overwrite"] is False
    assert calls[0]["progress_bar"] is True
    assert calls[0]["progress_every"] == 1000


def test_sample_no_progress_disables_runtime_progress_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
            "sample": {
                "seed": 1,
                "sequence_length": 12,
                "budget": {"tune": 1, "draws": 1},
                "optimizer": {"kind": "gibbs_anneal", "chains": 1, "cooling": {"kind": "fixed", "beta": 1.0}},
                "elites": {"k": 1},
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    calls: list[dict[str, object]] = []

    def _fake_run_sample(
        cfg,
        config_path_in: Path,
        *,
        force_overwrite: bool = False,
        progress_bar: bool = True,
        progress_every: int = 0,
    ) -> None:
        calls.append(
            {
                "config_path": config_path_in,
                "force_overwrite": force_overwrite,
                "progress_bar": progress_bar,
                "progress_every": progress_every,
            }
        )

    monkeypatch.setattr("dnadesign.cruncher.app.sample_workflow.run_sample", _fake_run_sample)

    result = invoke_cli(["sample", "--no-progress", "--verbose", str(config_path)])
    assert result.exit_code == 0
    assert calls
    assert calls[0]["config_path"] == config_path
    assert calls[0]["force_overwrite"] is False
    assert calls[0]["progress_bar"] is False
    assert calls[0]["progress_every"] == 0


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
    config_path = tmp_path / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
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
    result = invoke_cli(["config", "--config", str(CONFIG_PATH)])
    assert result.exit_code == 0
    assert "Cruncher config summary" in result.output


def test_config_summary_subcommand_defaults_to_resolved_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(CONFIG_PATH.parent)
    result = invoke_cli(["config", "summary"])
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

    result = invoke_cli(["config", "--config", str(config_path)])

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


def test_analyze_respects_analysis_enabled_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
            "analysis": {"enabled": False},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    called = {"value": False}

    def _boom(*args, **kwargs):
        called["value"] = True
        raise AssertionError("run_analyze should not be called when analysis.enabled is false")

    monkeypatch.setattr(analyze_workflow, "run_analyze", _boom)

    result = invoke_cli(["analyze", "--latest", str(config_path)])
    assert result.exit_code != 0
    assert "analysis.enabled=false" in result.output
    assert called["value"] is False


def test_discover_respects_discover_enabled_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
            "discover": {"enabled": False},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    called = {"value": False}

    def _boom(*args, **kwargs):
        called["value"] = True
        raise AssertionError("discover target resolution should not run when discover.enabled is false")

    monkeypatch.setattr(discover_module, "_resolve_targets", _boom)

    result = invoke_cli(["discover", "motifs", str(config_path)])
    assert result.exit_code != 0
    assert "discover.enabled=false" in result.output
    assert called["value"] is False


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


def test_lock_missing_catalog_data_hint_mentions_discovery_flow(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "outputs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = invoke_cli(["lock", str(config_path)])
    assert result.exit_code != 0
    assert "discover motifs" in combined_output(result)


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

    run_dir = tmp_path / "results"
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

    manifest_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(run_dir).write_text(
        json.dumps(
            {
                "stage": "sample",
                "run_dir": str(run_dir.resolve()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "draws": 1,
                "adapt_sweeps": 1,
                "top_k": 1,
                "objective": {"bidirectional": True, "score_scale": "normalized-llr"},
                "optimizer": {"kind": "gibbs_anneal"},
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
    assert "Use either --run or --latest, not both." in combined_output(result)


def test_analyze_allows_missing_analysis_section(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    captured: dict[str, object] = {}

    def _fake_run_analyze(cfg, config_path, *, runs_override=None, use_latest=False):
        captured["analysis_cfg"] = cfg.analysis
        captured["runs_override"] = runs_override
        captured["use_latest"] = use_latest
        return []

    monkeypatch.setattr(analyze_workflow, "run_analyze", _fake_run_analyze)

    result = invoke_cli(["analyze", "--latest", str(config_path)])

    assert result.exit_code == 0
    assert captured["analysis_cfg"] is None
    assert captured["runs_override"] is None
    assert captured["use_latest"] is True


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


def test_analyze_prints_next_steps_with_run_dir_not_analysis_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "outputs", "regulator_sets": [["lexA", "cpxR"]]},
            "catalog": {"root": str(tmp_path / ".cruncher"), "pwm_source": "matrix"},
            "sample": {
                "seed": 7,
                "sequence_length": 12,
                "budget": {"tune": 1, "draws": 1},
                "elites": {"k": 1},
                "output": {"save_sequences": True, "save_trace": False},
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = tmp_path / "outputs"
    analysis_dir = run_dir / "analysis"
    report_dir = analysis_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps({"analysis_id": "aid-1"}))
    (report_dir / "report.md").write_text("# report\n")

    def _fake_run_analyze(cfg, config_path, *, runs_override=None, use_latest=False):
        return [analysis_dir]

    monkeypatch.setattr(analyze_workflow, "run_analyze", _fake_run_analyze)

    result = invoke_cli(["analyze", "--latest", str(config_path)])

    assert result.exit_code == 0
    assert "cruncher runs show outputs -c" in result.output
    assert "cruncher notebook --latest" in result.output
    assert str(run_dir.resolve()) in result.output
