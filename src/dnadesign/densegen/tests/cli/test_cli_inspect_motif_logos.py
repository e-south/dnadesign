"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_inspect_motif_logos.py

CLI tests for read-only pwm_artifact logo rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.adapters.sources.pwm_sampling import build_log_odds
from dnadesign.densegen.src.cli.main import app

runner = CliRunner()


def _write_pwm_artifact(path: Path, *, motif_id: str, tf_name: str, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = [
        {"A": 0.70, "C": 0.10, "G": 0.10, "T": 0.10},
        {"A": 0.10, "C": 0.70, "G": 0.10, "T": 0.10},
        {"A": 0.10, "C": 0.10, "G": 0.70, "T": 0.10},
    ]
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    payload = {
        "schema_version": "1.0",
        "producer": "cruncher",
        "motif_id": motif_id,
        "tf_name": tf_name,
        "source": source,
        "alphabet": "ACGT",
        "matrix_semantics": "probabilities",
        "background": background,
        "probabilities": probabilities,
        "log_odds": build_log_odds(probabilities, background),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_config(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "inputs" / "motif_artifacts"
    _write_pwm_artifact(
        artifact_dir / "lexA__demo_source__lexA_CTGTATAWAWWHACA.json",
        motif_id="lexA_CTGTATAWAWWHACA",
        tf_name="lexA",
        source="demo_source",
    )
    _write_pwm_artifact(
        artifact_dir / "cpxR__demo_source__cpxR_MANWWHTTTAM.json",
        motif_id="cpxR_MANWWHTTTAM",
        tf_name="cpxR",
        source="demo_source",
    )
    manifest = {
        "schema_version": "1.0",
        "producer": "cruncher",
        "artifacts": [
            {
                "tf_name": "lexA",
                "source": "demo_source",
                "motif_id": "lexA_CTGTATAWAWWHACA",
                "path": "lexA__demo_source__lexA_CTGTATAWAWWHACA.json",
            },
            {
                "tf_name": "cpxR",
                "source": "demo_source",
                "motif_id": "cpxR_MANWWHTTTAM",
                "path": "cpxR__demo_source__cpxR_MANWWHTTTAM.json",
            },
        ],
    }
    (artifact_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo_pwm_artifacts
                root: "."
              inputs:
                - name: lexA_pwm
                  type: pwm_artifact
                  path: inputs/motif_artifacts/lexA__demo_source__lexA_CTGTATAWAWWHACA.json
                  sampling:
                    strategy: stochastic
                    n_sites: 2
                    mining:
                      batch_size: 10
                      budget:
                        mode: fixed_candidates
                        candidates: 100
                    length:
                      policy: exact
                - name: cpxR_pwm
                  type: pwm_artifact
                  path: inputs/motif_artifacts/cpxR__demo_source__cpxR_MANWWHTTTAM.json
                  sampling:
                    strategy: stochastic
                    n_sites: 2
                    mining:
                      batch_size: 10
                      budget:
                        mode: fixed_candidates
                        candidates: 100
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
                sequence_length: 30
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [lexA_pwm, cpxR_pwm]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )
    return cfg_path


def test_inspect_motif_logos_writes_png_and_svg_without_touching_inputs(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    artifact_dir = tmp_path / "inputs" / "motif_artifacts"
    artifact_before = (artifact_dir / "lexA__demo_source__lexA_CTGTATAWAWWHACA.json").read_text()
    manifest_before = (artifact_dir / "artifact_manifest.json").read_text()

    result = runner.invoke(
        app,
        ["inspect", "motif-logos", "-c", str(cfg_path)],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code == 0, result.output
    assert "Rendered motif logos" in result.output
    out_dir = tmp_path / "outputs" / "plots" / "motif_logos"
    assert (out_dir / "lexA__demo_source__lexA_CTGTATAWAWWHACA_logo.png").exists()
    assert (out_dir / "lexA__demo_source__lexA_CTGTATAWAWWHACA_logo.svg").exists()
    assert (out_dir / "cpxR__demo_source__cpxR_MANWWHTTTAM_logo.png").exists()
    assert (out_dir / "cpxR__demo_source__cpxR_MANWWHTTTAM_logo.svg").exists()
    assert (artifact_dir / "lexA__demo_source__lexA_CTGTATAWAWWHACA.json").read_text() == artifact_before
    assert (artifact_dir / "artifact_manifest.json").read_text() == manifest_before


def test_inspect_motif_logos_supports_input_filter(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)

    result = runner.invoke(
        app,
        ["inspect", "motif-logos", "-c", str(cfg_path), "--input", "lexA_pwm"],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "outputs" / "plots" / "motif_logos"
    assert (out_dir / "lexA__demo_source__lexA_CTGTATAWAWWHACA_logo.svg").exists()
    assert not (out_dir / "cpxR__demo_source__cpxR_MANWWHTTTAM_logo.svg").exists()


def test_inspect_motif_logos_rejects_workspace_local_out_dir_outside_outputs(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)

    result = runner.invoke(
        app,
        ["inspect", "motif-logos", "-c", str(cfg_path), "--out-dir", "inputs/logo_exports"],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code == 1
    assert "must not be under inputs/" in result.output
