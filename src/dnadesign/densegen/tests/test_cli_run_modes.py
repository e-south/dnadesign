"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_cli_run_modes.py

CLI run-mode guardrail tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app
from dnadesign.densegen.src.cli_commands import run as run_command
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
              path: outputs/tables/dense_arrays.parquet
          generation:
            sequence_length: 10
            quota: 1
            plan:
              - name: demo_plan
                quota: 1
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
              path: outputs/tables/dense_arrays.parquet
          generation:
            sequence_length: 10
            quota: 1
            plan:
              - name: demo_plan
                quota: 1
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


def test_run_auto_resumes_when_outputs_exist_and_pools_present(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    outputs_dir = run_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "dense_arrays.parquet").write_text("seed")
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
    (tables_dir / "dense_arrays.parquet").write_text("seed")

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
    (tables_dir / "dense_arrays.parquet").write_text("seed")

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
    (outputs_dir / "tables" / "dense_arrays.parquet").write_text("seed")

    runner = CliRunner()
    result = runner.invoke(app, ["campaign-reset", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert not outputs_dir.exists()
    assert (run_root / "inputs.csv").exists()


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
