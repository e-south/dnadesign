from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app


def _write_config(run_root: Path) -> Path:
    cfg_path = run_root / "config.yaml"
    cfg_path.write_text(
        """
        densegen:
          schema_version: "2.8"
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
          schema_version: "2.8"
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


def test_run_requires_explicit_mode_when_outputs_exist(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    outputs_dir = run_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "existing.txt").write_text("seed")

    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path)])

    assert result.exit_code != 0, result.output
    assert "Existing outputs found" in result.output


def test_run_resume_requires_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _write_inputs(run_root)
    cfg_path = _write_config(run_root)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "-c", str(cfg_path), "--resume"])

    assert result.exit_code != 0, result.output
    assert "--resume requested but no outputs were found" in result.output


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


def test_run_requires_stage_a_pool_when_pwm_inputs_present(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    (run_root / "pwm.csv").write_text("A,C,G,T\n0.25,0.25,0.25,0.25\n")
    cfg_path = _write_pwm_config(run_root)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--fresh", "-c", str(cfg_path)])

    assert result.exit_code != 0, result.output
    assert "Stage-A pools" in result.output
    assert "stage-a build-pool" in result.output
