"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_cli_stage_b_errors.py

CLI coverage for Stage-B build-libraries error messaging.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app


def _write_stage_b_config(
    tmp_path: Path,
    *,
    required_regulators: list[str],
    library_sampling_strategy: str = "tf_balanced",
) -> Path:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    sites_path = inputs_dir / "sites.csv"
    sites_path.write_text("tf,tfbs\nTF_A,AAAA\nTF_B,CCCC\n")
    cfg_path = tmp_path / "config.yaml"
    required = ", ".join(f"{reg!r}" for reg in required_regulators)
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            densegen:
              schema_version: "2.5"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: {sites_path}
                  format: csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/dense_arrays.parquet
              generation:
                sequence_length: 30
                quota: 1
                sampling:
                  pool_strategy: subsample
                  library_size: 2
                  library_sampling_strategy: {library_sampling_strategy}
                  cover_all_regulators: true
                plan:
                  - name: default
                    quota: 1
                    required_regulators: [{required}]
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


def _write_pool_manifest(tmp_path: Path) -> Path:
    pools_dir = tmp_path / "outputs" / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "input_name": ["demo_input"] * 2,
            "tf": ["TF_A", "TF_B"],
            "tfbs": ["AAAA", "CCCC"],
            "best_hit_score": [5.0, 4.0],
            "tier": [0, 1],
            "rank_within_regulator": [1, 1],
        }
    )
    pool_path = pools_dir / "demo_input__pool.parquet"
    df.to_parquet(pool_path, index=False)
    manifest = {
        "schema_version": "1.3",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "demo_input",
                "type": "binding_sites",
                "pool_path": "demo_input__pool.parquet",
                "rows": int(len(df)),
                "columns": list(df.columns),
                "pool_mode": "tfbs",
            }
        ],
    }
    (pools_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))
    return pools_dir


def test_stage_b_reports_missing_required_regulators(tmp_path: Path) -> None:
    cfg_path = _write_stage_b_config(tmp_path, required_regulators=["MISSING_TF"])
    pool_dir = _write_pool_manifest(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "stage-b",
            "build-libraries",
            "-c",
            str(cfg_path),
            "--pool",
            str(pool_dir),
        ],
    )
    assert result.exit_code != 0, result.output
    assert "Stage-B sampling failed" in result.output
    assert "Required regulators not found in input" in result.output
    assert "Available regulators" in result.output


def test_stage_b_emits_sampling_pressure_events(tmp_path: Path) -> None:
    cfg_path = _write_stage_b_config(
        tmp_path,
        required_regulators=["TF_A", "TF_B"],
        library_sampling_strategy="coverage_weighted",
    )
    pool_dir = _write_pool_manifest(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "stage-b",
            "build-libraries",
            "-c",
            str(cfg_path),
            "--pool",
            str(pool_dir),
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output
    events_path = tmp_path / "outputs" / "meta" / "events.jsonl"
    assert events_path.exists()
    rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert any(row.get("event") == "LIBRARY_SAMPLING_PRESSURE" for row in rows)


def test_stage_b_requires_overwrite_when_artifacts_exist(tmp_path: Path) -> None:
    cfg_path = _write_stage_b_config(
        tmp_path,
        required_regulators=["TF_A", "TF_B"],
        library_sampling_strategy="coverage_weighted",
    )
    pool_dir = _write_pool_manifest(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "stage-b",
            "build-libraries",
            "-c",
            str(cfg_path),
            "--pool",
            str(pool_dir),
            "--overwrite",
        ],
    )
    assert first.exit_code == 0, first.output
    second = runner.invoke(
        app,
        [
            "stage-b",
            "build-libraries",
            "-c",
            str(cfg_path),
            "--pool",
            str(pool_dir),
        ],
    )
    assert second.exit_code != 0, second.output
    assert "overwrite" in second.output.lower()
