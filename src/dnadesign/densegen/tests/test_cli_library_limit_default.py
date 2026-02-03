"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_cli_library_limit_default.py

CLI summary aggregation tests for library listings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app
from dnadesign.densegen.tests.config_fixtures import write_minimal_config

PLAN_POOL_LABEL = "plan_pool__demo_plan"


def test_inspect_run_library_summary_is_aggregated(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\nTF1,AAA\n")

    tables_root = tmp_path / "outputs" / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    meta_root = tmp_path / "outputs" / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_id": "demo",
        "created_at": "2026-01-14T00:00:00+00:00",
        "schema_version": "2.9",
        "config_sha256": "dummy",
        "run_root": ".",
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
                "generated": 0,
                "duplicates_skipped": 0,
                "failed_solutions": 0,
                "total_resamples": 0,
                "libraries_built": 0,
                "stall_events": 0,
            }
        ],
    }
    (meta_root / "run_manifest.json").write_text(json.dumps(run_manifest))
    attempts_rows = []
    hashes = []
    for idx in range(1, 4):
        lib_hash = f"hash{idx:02d}"
        hashes.append(lib_hash)
        attempts_rows.append(
            {
                "attempt_id": f"a{idx}",
                "attempt_index": idx,
                "run_id": "demo",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:00+00:00",
                "status": "success",
                "sampling_library_index": idx,
                "sampling_library_hash": lib_hash,
                "library_tfbs": ["AAA"],
                "library_tfs": ["TF1"],
                "library_site_ids": [f"s{idx}"],
                "library_sources": ["demo"],
            }
        )
    pd.DataFrame(attempts_rows).to_parquet(tables_root / "attempts.parquet", index=False)

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", "--library", "-c", str(cfg_path)])
    assert result.exit_code == 0
    assert "TF usage summary" in result.output
    assert "Library build summary" not in result.output
    for lib_hash in hashes:
        assert lib_hash not in result.output
