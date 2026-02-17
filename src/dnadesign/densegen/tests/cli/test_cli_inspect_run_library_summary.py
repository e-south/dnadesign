"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_inspect_run_library_summary.py

CLI tests for inspect run library summaries without TFBS sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.densegen.src.cli.main import app
from dnadesign.densegen.tests.config_fixtures import write_minimal_config

PLAN_POOL_LABEL = "plan_pool__demo_plan"


def _write_attempts(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "attempt_id": "a1",
                "attempt_index": 1,
                "run_id": "demo",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:01+00:00",
                "status": "success",
                "reason": "ok",
                "detail_json": "{}",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash1",
                "solution_id": "out1",
                "used_tf_counts_json": "{}",
                "used_tf_list": ["TF1", "TF2"],
                "sampling_library_index": 1,
                "sampling_library_hash": "abc123",
                "solver_status": "optimal",
                "solver_objective": 0.0,
                "solver_solve_time_s": 0.1,
                "dense_arrays_version": None,
                "library_tfbs": ["AAA", "CCC"],
                "library_tfs": ["TF1", "TF2"],
                "library_site_ids": ["s1", "s2"],
                "library_sources": ["demo", "demo"],
            }
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_inspect_run_library_summary_hides_tfbs_sequences(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\nTF1,AAA\nTF2,CCC\n")
    _write_attempts(tmp_path / "outputs" / "tables" / "attempts.parquet")
    meta_root = tmp_path / "outputs" / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_id": "demo",
        "created_at": "2026-01-14T00:00:00+00:00",
        "schema_version": "2.9",
        "config_sha256": "dummy",
        "run_root": str(tmp_path),
        "random_seed": 0,
        "seed_stage_a": 0,
        "seed_stage_b": 0,
        "seed_solver": 0,
        "solver_backend": "CBC",
        "solver_strategy": "iterate",
        "solver_time_limit_seconds": None,
        "solver_threads": None,
        "solver_strands": "double",
        "dense_arrays_version": None,
        "dense_arrays_version_source": "unknown",
        "items": [
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "generated": 1,
                "duplicates_skipped": 0,
                "failed_solutions": 0,
                "total_resamples": 0,
                "libraries_built": 1,
                "stall_events": 0,
            }
        ],
    }
    (meta_root / "run_manifest.json").write_text(json.dumps(run_manifest))

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", "--library", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "TF usage summary" in result.output
    assert "TFBS usage summary" in result.output
    assert "AAA" not in result.output
