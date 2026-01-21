from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app


def _write_config(run_root: Path) -> Path:
    cfg_path = run_root / "config.yaml"
    cfg_path.write_text(
        """
        densegen:
          schema_version: "2.4"
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
              path: outputs/dense_arrays.parquet
          generation:
            sequence_length: 10
            quota: 1
            plan:
              - name: demo_plan
                quota: 1
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )
    return cfg_path


def _write_inputs(run_root: Path) -> None:
    (run_root / "inputs.csv").write_text("tf,sequence\nlexA,ATGC\n")


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
