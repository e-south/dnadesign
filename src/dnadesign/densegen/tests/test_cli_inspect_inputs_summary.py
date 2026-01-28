from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app


def _write_config(path: Path) -> None:
    path.write_text(
        """
        densegen:
          schema_version: "2.5"
          run:
            id: demo
            root: "."
          inputs:
            - name: demo_sites
              type: binding_sites
              path: inputs.csv
            - name: demo_pwm
              type: pwm_matrix_csv
              path: pwm.csv
              motif_id: demo
              sampling:
                strategy: stochastic
                scoring_backend: densegen
                n_sites: 2
                score_threshold: 0.0
                oversample_factor: 2
                max_candidates: 50
                max_seconds: 1
                length_policy: exact
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
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )


def test_inspect_inputs_uses_clear_labels(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")
    (tmp_path / "pwm.csv").write_text("A,C,G,T\n0.25,0.25,0.25,0.25\n")

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "inputs", "-c", str(cfg_path)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "inputs" in result.output
    assert "file=" in result.output
    assert "candidate_cap" in result.output
    assert "time_cap_s" in result.output
