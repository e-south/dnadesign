"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_run_modes.py

CLI run-mode guardrail tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

from dnadesign.densegen.src.cli import run as run_command
from dnadesign.densegen.src.cli.main import app
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.artifacts.pool import (
    POOL_SCHEMA_VERSION,
    _hash_pool_config,
    _resolve_input_fingerprints,
)


def _write_config(run_root: Path) -> Path:
    cfg_path = run_root / "config.yaml"
    cfg_path.write_text(
        """
        densegen:
          schema_version: "2.9"
          run:
            id: demo
            root: "."
          inputs:
            - name: demo_input
              type: binding_sites
              path: inputs.csv
          output:
            targets: [parquet]
            schema:
              bio_type: dna
              alphabet: dna_4
            parquet:
              path: outputs/tables/records.parquet
          generation:
            sequence_length: 10
            plan:
              - name: demo_plan
                sequences: 1
                sampling:
                  include_inputs: [demo_input]
                regulator_constraints:
                  groups:
                    - name: all
                      members: [lexA]
                      min_required: 1
          solver:
            backend: CBC
            strategy: iterate
          postprocess:
            pad:
              mode: adaptive
              end: 5prime
              gc:
                mode: range
                min: 0.4
                max: 0.6
                target: 0.5
                tolerance: 0.1
                min_pad_length: 4
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )
    return cfg_path


def _write_many_plan_config(run_root: Path, *, plan_count: int, quota: int = 1) -> Path:
    if plan_count <= 0:
        raise ValueError("plan_count must be positive")
    cfg_path = run_root / "config.yaml"
    plan_items = []
    for idx in range(plan_count):
        plan_name = f"demo_plan_{idx + 1}"
        plan_items.append(
            {
                "name": plan_name,
                "sequences": int(quota),
                "sampling": {"include_inputs": ["demo_input"]},
                "regulator_constraints": {"groups": [{"name": "all", "members": ["lexA"], "min_required": 1}]},
            }
        )
    payload = {
        "densegen": {
            "schema_version": "2.9",
            "run": {"id": "demo", "root": "."},
            "inputs": [{"name": "demo_input", "type": "binding_sites", "path": "inputs.csv"}],
            "output": {
                "targets": ["parquet"],
                "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                "parquet": {"path": "outputs/tables/records.parquet"},
            },
            "generation": {"sequence_length": 10, "plan": plan_items},
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "postprocess": {
                "pad": {
                    "mode": "adaptive",
                    "end": "5prime",
                    "gc": {
                        "mode": "range",
                        "min": 0.4,
                        "max": 0.6,
                        "target": 0.5,
                        "tolerance": 0.1,
                        "min_pad_length": 4,
                    },
                }
            },
            "logging": {"log_dir": "outputs/logs"},
        }
    }
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return cfg_path


def _write_config_with_auxiliary_input(run_root: Path) -> Path:
    cfg_path = run_root / "config.yaml"
    cfg_path.write_text(
        """
        densegen:
          schema_version: "2.9"
          run:
            id: demo
            root: "."
          inputs:
            - name: demo_input
              type: binding_sites
              path: inputs.csv
            - name: aux_input
              type: binding_sites
              path: missing.csv
          output:
            targets: [parquet]
            schema:
              bio_type: dna
              alphabet: dna_4
            parquet:
              path: outputs/tables/records.parquet
          generation:
            sequence_length: 10
            plan:
              - name: demo_plan
                sequences: 1
                sampling:
                  include_inputs: [demo_input]
                regulator_constraints:
                  groups:
                    - name: all
                      members: [lexA]
                      min_required: 1
          solver:
            backend: CBC
            strategy: iterate
          postprocess:
            pad:
              mode: adaptive
              end: 5prime
              gc:
                mode: range
                min: 0.4
                max: 0.6
                target: 0.5
                tolerance: 0.1
                min_pad_length: 4
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )
    return cfg_path


def _write_screen_progress_config(run_root: Path) -> Path:
    cfg_path = _write_config(run_root)
    marker = "          logging:\n            log_dir: outputs/logs\n"
    replacement = "          logging:\n            log_dir: outputs/logs\n            progress_style: screen\n"
    text = cfg_path.read_text()
    assert marker in text
    cfg_path.write_text(text.replace(marker, replacement, 1))
    return cfg_path


def _write_auto_progress_config(run_root: Path) -> Path:
    cfg_path = _write_config(run_root)
    marker = "          logging:\n            log_dir: outputs/logs\n"
    replacement = "          logging:\n            log_dir: outputs/logs\n            progress_style: auto\n"
    text = cfg_path.read_text()
    assert marker in text
    cfg_path.write_text(text.replace(marker, replacement, 1))
    return cfg_path


def _write_inputs(run_root: Path) -> None:
    (run_root / "inputs.csv").write_text("tf,sequence\nlexA,ATGC\n")


def _write_pwm_config(run_root: Path) -> Path:
    cfg_path = run_root / "config.yaml"
    cfg_path.write_text(
        """
        densegen:
          schema_version: "2.9"
          run:
            id: demo
            root: "."
          inputs:
            - name: demo_pwm
              type: pwm_matrix_csv
              path: pwm.csv
              motif_id: demo_motif
              sampling:
                strategy: stochastic
                n_sites: 2
                mining:
                  batch_size: 10
                  budget:
                    mode: fixed_candidates
                    candidates: 20
                length:
                  policy: exact
          output:
            targets: [parquet]
            schema:
              bio_type: dna
              alphabet: dna_4
            parquet:
              path: outputs/tables/records.parquet
          generation:
            sequence_length: 10
            plan:
              - name: demo_plan
                sequences: 1
                sampling:
                  include_inputs: [demo_pwm]
                regulator_constraints:
                  groups:
                    - name: all
                      members: [demo_motif]
                      min_required: 1
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )
    return cfg_path


def _write_pool_manifest(run_root: Path, cfg_path: Path) -> None:
    loaded = load_config(cfg_path)
    cfg = loaded.root.densegen
    pool_dir = run_root / "outputs" / "pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for inp in cfg.inputs:
        pool_path = f"{inp.name}__pool.parquet"
        (pool_dir / pool_path).write_text("seed")
        entries.append(
            {
                "name": inp.name,
                "type": inp.type,
                "pool_path": pool_path,
                "rows": 1,
                "columns": ["tf", "tfbs", "motif_id", "tfbs_id"],
                "pool_mode": "tfbs",
                "fingerprints": _resolve_input_fingerprints(cfg_path, inp),
            }
        )
    payload = {
        "schema_version": POOL_SCHEMA_VERSION,
        "run_id": cfg.run.id,
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "config_hash": _hash_pool_config(cfg),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": entries,
    }
    (pool_dir / "pool_manifest.json").write_text(json.dumps(payload))


def _write_usr_config(run_root: Path) -> Path:
    cfg_path = run_root / "config.yaml"
    cfg_path.write_text(
        """
        densegen:
          schema_version: "2.9"
          run:
            id: demo
            root: "."
          inputs:
            - name: demo_input
              type: binding_sites
              path: inputs.csv
          output:
            targets: [usr]
            schema:
              bio_type: dna
              alphabet: dna_4
            usr:
              root: outputs/usr_datasets
              dataset: demo
              chunk_size: 16
          generation:
            sequence_length: 10
            plan:
              - name: demo_plan
                sequences: 1
                sampling:
                  include_inputs: [demo_input]
                regulator_constraints:
                  groups:
                    - name: all
                      members: [lexA]
                      min_required: 1
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )
    return cfg_path


def test_run_auto_resumes_when_outputs_exist_and_pools_present(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    outputs_dir = run_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "records.parquet").write_text("seed")
    _write_pool_manifest(run_root, cfg_path)

    captured: dict[str, bool] = {}

    def _fake_run_pipeline(_loaded, *, resume, build_stage_a, **_kwargs):
        captured["resume"] = bool(resume)
        captured["build_stage_a"] = bool(build_stage_a)

    runner = CliRunner()
    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    result = runner.invoke(app, ["run", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert captured["resume"] is True
    assert captured["build_stage_a"] is False


def test_run_next_steps_use_plot_discovery_and_generic_plot_command(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    monkeypatch.setattr(run_command, "run_pipeline", lambda *_args, **_kwargs: None)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--fresh", "--no-plot", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    normalized = result.output.replace("\n", " ")
    assert "dense ls-plots" in normalized
    assert "dense plot --only stage_a_summary" not in normalized
    assert "render configured plot set" in normalized


def test_help_lists_campaign_reset_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "campaign-reset" in result.output


def test_run_resume_requires_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--resume"])

    assert result.exit_code != 0, result.output
    assert "--resume requested but no outputs were found" in result.output


def test_run_reports_run_state_config_mismatch(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)
    outputs_dir = run_root / "outputs"
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "records.parquet").write_text("seed")

    def _fake_run_pipeline(*_args, **_kwargs):
        raise RuntimeError(
            "Existing run_state.json was created with a different config. "
            "Remove run_state.json or stage a new run root to start fresh."
        )

    runner = CliRunner()
    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--resume"])

    assert result.exit_code != 0, result.output
    normalized = result.output.replace("\n", " ")
    assert "run_state.json was created with a different config" in normalized
    assert "run --fresh" in normalized
    assert "campaign-reset" in normalized


def test_run_auto_builds_stage_a_when_pools_missing(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    outputs_dir = run_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "records.parquet").write_text("seed")

    captured: dict[str, bool] = {}

    def _fake_run_pipeline(_loaded, *, resume, build_stage_a, **_kwargs):
        captured["resume"] = bool(resume)
        captured["build_stage_a"] = bool(build_stage_a)

    runner = CliRunner()
    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    result = runner.invoke(app, ["run", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert captured["resume"] is True
    assert captured["build_stage_a"] is True


def test_run_ignores_stale_auxiliary_inputs_not_in_active_plan(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config_with_auxiliary_input(run_root)

    outputs_dir = run_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "records.parquet").write_text("seed")

    captured: dict[str, bool] = {}

    def _fake_run_pipeline(_loaded, *, resume, build_stage_a, **_kwargs):
        captured["resume"] = bool(resume)
        captured["build_stage_a"] = bool(build_stage_a)

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(
        run_command,
        "pool_status_by_input",
        lambda *_args, **_kwargs: {
            "demo_input": SimpleNamespace(name="demo_input", state="present"),
            "aux_input": SimpleNamespace(name="aux_input", state="stale"),
        },
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert captured["resume"] is True
    assert captured["build_stage_a"] is False
    assert "Ignoring stale Stage-A pools for unused inputs: aux_input" in result.output


def test_campaign_reset_removes_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    outputs_dir = run_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "meta").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "meta" / "run_state.json").write_text("{}")
    (outputs_dir / "tables" / "records.parquet").write_text("seed")

    runner = CliRunner()
    result = runner.invoke(app, ["campaign-reset", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert not outputs_dir.exists()
    assert (run_root / "inputs.csv").exists()


def test_run_fails_fast_for_screen_progress_without_interactive_terminal(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_screen_progress_config(run_root)
    outputs_dir = run_root / "outputs"
    sentinel = outputs_dir / "tables" / "records.parquet"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("seed")

    called = {"run_pipeline": False}

    def _fake_run_pipeline(*_args, **_kwargs):
        called["run_pipeline"] = True

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--fresh", "-c", str(cfg_path)])

    assert result.exit_code != 0, result.output
    assert "interactive terminal" in result.output
    assert "progress_style: stream" in result.output
    assert called["run_pipeline"] is False
    assert sentinel.exists()


def test_run_auto_progress_downgrades_to_summary_for_non_interactive_stdout(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_auto_progress_config(run_root)

    captured: dict[str, str] = {}

    def _fake_run_pipeline(loaded, **_kwargs):
        captured["style"] = str(loaded.root.densegen.logging.progress_style)

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--fresh", "--no-plot", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert captured["style"] == "summary"
    assert "progress_style=auto -> summary" in result.output


def test_run_auto_progress_prefers_non_interactive_summary_over_dumb_term(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_auto_progress_config(run_root)

    captured: dict[str, str] = {}

    def _fake_run_pipeline(loaded, **_kwargs):
        captured["style"] = str(loaded.root.densegen.logging.progress_style)

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setenv("TERM", "dumb")
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--fresh", "--no-plot", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert captured["style"] == "summary"
    assert "progress_style=auto -> summary" in result.output


def test_run_handles_screen_progress_runtime_errors_with_actionable_next_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    def _fake_run_pipeline(*_args, **_kwargs):
        raise RuntimeError(
            "logging.progress_style=screen requires an interactive terminal. "
            "Use logging.progress_style=stream for non-interactive output."
        )

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path)])

    assert result.exit_code != 0, result.output
    assert "interactive terminal" in result.output
    assert "Next steps" in result.output
    assert "progress_style: stream" in result.output


def test_run_handles_max_consecutive_no_progress_runtime_errors_with_actionable_next_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    def _fake_run_pipeline(*_args, **_kwargs):
        raise RuntimeError("[plan_pool__demo/demo_plan] Exceeded max_consecutive_no_progress_resamples=120.")

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path)])

    assert result.exit_code != 0, result.output
    normalized = result.output.replace("\n", " ")
    assert "Exceeded max_consecutive_no_progress_resamples=120" in normalized
    assert "inspect run --events --library" in normalized
    assert "increase densegen.runtime.no_progress_seconds_before_resample" in normalized
    assert "Traceback" not in result.output


def test_run_handles_missing_usr_registry_runtime_errors_with_actionable_next_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    def _fake_run_pipeline(*_args, **_kwargs):
        raise RuntimeError(
            "USR registry not found at /tmp/demo/outputs/usr_datasets/registry.yaml. "
            "Create registry.yaml before writing USR outputs."
        )

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path)])

    assert result.exit_code != 0, result.output
    normalized = result.output.replace("\n", " ")
    assert "USR registry not found" in normalized
    assert "workspace init --output-mode usr|both" in normalized
    assert "registry.yaml" in normalized
    assert "Traceback" not in result.output


def test_run_fresh_rebuilds_stage_a(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    captured: dict[str, bool] = {}

    def _fake_run_pipeline(_loaded, *, resume, build_stage_a, **_kwargs):
        captured["resume"] = bool(resume)
        captured["build_stage_a"] = bool(build_stage_a)

    runner = CliRunner()
    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    result = runner.invoke(app, ["run", "--fresh", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert captured["resume"] is False
    assert captured["build_stage_a"] is True


def test_run_fresh_preserves_usr_registry(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_usr_config(run_root)
    registry_path = run_root / "outputs" / "usr_datasets" / "registry.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("namespaces: {}\n")

    def _fake_run_pipeline(_loaded, *, resume, build_stage_a, **_kwargs):
        return None

    runner = CliRunner()
    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    result = runner.invoke(app, ["run", "--fresh", "-c", str(cfg_path), "--no-plot"])

    assert result.exit_code == 0, result.output
    assert registry_path.exists()
    assert registry_path.read_text() == "namespaces: {}\n"


def test_run_fresh_preserves_notify_profile_and_cursor(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)
    notify_dir = run_root / "outputs" / "notify" / "densegen"
    notify_dir.mkdir(parents=True, exist_ok=True)
    profile_path = notify_dir / "profile.json"
    cursor_path = notify_dir / "cursor"
    profile_path.write_text('{"provider":"slack"}\n')
    cursor_path.write_text("12345\n")
    stale_tables = run_root / "outputs" / "tables"
    stale_tables.mkdir(parents=True, exist_ok=True)
    stale_marker = stale_tables / "records.parquet"
    stale_marker.write_text("stale\n")

    def _fake_run_pipeline(_loaded, *, resume, build_stage_a, **_kwargs):
        return None

    runner = CliRunner()
    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    result = runner.invoke(app, ["run", "--fresh", "-c", str(cfg_path), "--no-plot"])

    assert result.exit_code == 0, result.output
    assert profile_path.exists()
    assert profile_path.read_text() == '{"provider":"slack"}\n'
    assert cursor_path.exists()
    assert cursor_path.read_text() == "12345\n"
    assert not stale_marker.exists()


def test_run_auto_accepts_plan_quota_increase_on_resume(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)
    old_cfg = load_config(cfg_path).root.densegen.model_dump(by_alias=True, exclude_none=False)
    old_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()

    updated_text = cfg_path.read_text().replace("                sequences: 1\n", "                sequences: 2\n", 1)
    cfg_path.write_text(updated_text)
    outputs_dir = run_root / "outputs"
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "tables" / "records.parquet").write_text("seed")
    (outputs_dir / "meta").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "meta" / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": "demo",
                "created_at": "2026-02-01T00:00:00Z",
                "updated_at": "2026-02-01T00:00:00Z",
                "schema_version": "2.9",
                "config_sha256": old_sha,
                "run_root": ".",
                "items": [{"input_name": "demo_input", "plan_name": "demo_plan", "generated": 1}],
            }
        )
    )
    (outputs_dir / "meta" / "effective_config.json").write_text(
        json.dumps(
            {
                "config": old_cfg,
                "run_id": "demo",
            }
        )
    )

    captured: dict[str, bool] = {}

    def _fake_run_pipeline(_loaded, *, resume, build_stage_a, allow_config_mismatch=False, **_kwargs):
        captured["resume"] = bool(resume)
        captured["allow_config_mismatch"] = bool(allow_config_mismatch)
        return None

    runner = CliRunner()
    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--resume", "--no-plot"])

    assert result.exit_code == 0, result.output
    assert captured["resume"] is True
    assert captured["allow_config_mismatch"] is True


def test_run_reports_quota_already_met_when_resume_has_no_work(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    def _fake_run_pipeline(*_args, **_kwargs):
        return SimpleNamespace(total_generated=1, per_plan={("demo_input", "demo_plan"): 1}, generated_this_run=0)

    runner = CliRunner()
    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--resume", "--no-plot"])

    assert result.exit_code != 0, result.output
    assert "--resume requested but no outputs were found" in result.output

    outputs_dir = run_root / "outputs"
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "tables" / "records.parquet").write_text("seed")
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--resume", "--no-plot"])
    assert result.exit_code == 0, result.output
    assert "quota is already met" in result.output.lower()


def test_run_extend_quota_increases_plan_target_before_pipeline(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    captured: dict[str, int] = {}

    def _fake_run_pipeline(loaded, **_kwargs):
        captured["quota"] = int(loaded.root.densegen.generation.plan[0].sequences)
        return SimpleNamespace(total_generated=4, per_plan={("demo_input", "demo_plan"): 4}, generated_this_run=4)

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--extend-quota", "3", "--no-plot"])

    assert result.exit_code == 0, result.output
    assert captured["quota"] == 4
    assert "Quota plan" in result.output
    assert "demo_plan=4" in result.output


def test_run_quota_plan_output_is_grouped_for_large_plan_sets(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_many_plan_config(run_root, plan_count=9, quota=1)

    def _fake_run_pipeline(*_args, **_kwargs):
        return SimpleNamespace(total_generated=9, per_plan={}, generated_this_run=9)

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--fresh", "--no-plot"])

    assert result.exit_code == 0, result.output
    assert "Quota plan" in result.output
    assert "9 plans (quota pattern: 9 plans at 1 each)" in result.output
    assert "demo_plan_9=1" not in result.output


def test_run_extend_quota_rejects_non_positive_values(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--extend-quota", "0", "--no-plot"])

    assert result.exit_code == 1, result.output
    assert "--extend-quota must be > 0" in result.output


def test_run_extend_quota_anchors_to_existing_generated_rows(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)
    outputs_dir = run_root / "outputs"
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "tables" / "records.parquet").write_text("seed")
    (outputs_dir / "meta").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "meta" / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": "demo",
                "created_at": "2026-02-01T00:00:00Z",
                "updated_at": "2026-02-01T00:00:00Z",
                "schema_version": "2.9",
                "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                "run_root": ".",
                "items": [{"input_name": "demo_input", "plan_name": "demo_plan", "generated": 4}],
            }
        )
    )

    captured: dict[str, int] = {}

    def _fake_run_pipeline(loaded, **_kwargs):
        captured["quota"] = int(loaded.root.densegen.generation.plan[0].sequences)
        return SimpleNamespace(total_generated=7, per_plan={("demo_input", "demo_plan"): 7}, generated_this_run=3)

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--resume", "--extend-quota", "3", "--no-plot"])

    assert result.exit_code == 0, result.output
    assert captured["quota"] == 7


def test_run_resume_reuses_effective_quota_target_after_interrupted_extend(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    outputs_dir = run_root / "outputs"
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "tables" / "records.parquet").write_text("seed")
    (outputs_dir / "meta").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "meta" / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": "demo",
                "created_at": "2026-02-01T00:00:00Z",
                "updated_at": "2026-02-01T00:00:00Z",
                "schema_version": "2.9",
                "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                "run_root": ".",
                "items": [{"input_name": "demo_input", "plan_name": "demo_plan", "generated": 2}],
            }
        )
    )

    previous_cfg = load_config(cfg_path).root.densegen.model_dump(by_alias=True, exclude_none=False)
    previous_cfg["generation"]["plan"][0]["sequences"] = 6
    (outputs_dir / "meta" / "effective_config.json").write_text(
        json.dumps(
            {
                "config": previous_cfg,
                "run_id": "demo",
            }
        )
    )

    captured: dict[str, int] = {}

    def _fake_run_pipeline(loaded, **_kwargs):
        captured["quota"] = int(loaded.root.densegen.generation.plan[0].sequences)
        return SimpleNamespace(total_generated=6, per_plan={("demo_input", "demo_plan"): 6}, generated_this_run=4)

    monkeypatch.setattr(run_command, "run_pipeline", _fake_run_pipeline)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--resume", "--no-plot"])

    assert result.exit_code == 0, result.output
    assert captured["quota"] == 6
