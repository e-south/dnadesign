from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.densegen.src.adapters.outputs import OutputRecord, ParquetSink
from dnadesign.densegen.src.cli import app
from dnadesign.densegen.src.core.run_manifest import PlanManifest, RunManifest
from dnadesign.densegen.src.core.run_paths import ensure_run_meta_dir, run_manifest_path
from dnadesign.densegen.tests.meta_fixtures import output_meta


def _write_config(path: Path) -> None:
    path.write_text(
        """
        densegen:
          schema_version: "2.5"
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
          solver:
            backend: CBC
            strategy: iterate
          logging:
            log_dir: outputs/logs
        """.strip()
        + "\n"
    )


def test_summarize_library_grouping(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path)

    # outputs
    out_file = run_root / "outputs" / "tables" / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), chunk_size=1)
    meta = output_meta(library_hash="abc123", library_index=1)
    rec = OutputRecord.from_sequence(
        sequence="ATGCATGCAT",
        meta=meta,
        source="densegen:demo",
        bio_type="dna",
        alphabet="dna_4",
    )
    sink.add(rec)
    sink.finalize()

    # attempts parquet (library offered to solver)
    outputs_dir = run_root / "outputs" / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    attempts_df = pd.DataFrame(
        [
            {
                "attempt_id": "a1",
                "attempt_index": 1,
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:01+00:00",
                "status": "success",
                "reason": "ok",
                "detail_json": "{}",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash",
                "solution_id": "out1",
                "used_tf_counts_json": "{}",
                "used_tf_list": ["lexA", "cpxR"],
                "sampling_library_index": 1,
                "sampling_library_hash": "abc123",
                "solver_status": "optimal",
                "solver_objective": 0.0,
                "solver_solve_time_s": 0.1,
                "dense_arrays_version": None,
                "dense_arrays_version_source": "unknown",
                "library_tfbs": ["AAA", "CCC"],
                "library_tfs": ["lexA", "cpxR"],
                "library_site_ids": ["", ""],
                "library_sources": ["inputs.csv", "inputs.csv"],
            }
        ]
    )
    attempts_df.to_parquet(outputs_dir / "attempts.parquet", index=False)

    solutions_df = pd.DataFrame(
        [
            {
                "solution_id": "out1",
                "attempt_id": "a1",
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:01+00:00",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash",
                "sampling_library_index": 1,
                "sampling_library_hash": "abc123",
            }
        ]
    )
    solutions_df.to_parquet(outputs_dir / "solutions.parquet", index=False)

    # run manifest
    manifest = RunManifest(
        run_id="demo",
        created_at="2026-01-14T00:00:00+00:00",
        schema_version="2.5",
        config_sha256="dummy",
        run_root=str(run_root),
        random_seed=123,
        seed_stage_a=456,
        seed_stage_b=789,
        seed_solver=101112,
        solver_backend="CBC",
        solver_strategy="iterate",
        solver_time_limit_seconds=None,
        solver_threads=None,
        solver_strands="double",
        dense_arrays_version=None,
        dense_arrays_version_source="unknown",
        items=[
            PlanManifest(
                input_name="demo_input",
                plan_name="demo_plan",
                generated=1,
                duplicates_skipped=0,
                failed_solutions=0,
                total_resamples=0,
                libraries_built=1,
                stall_events=0,
            )
        ],
    )
    ensure_run_meta_dir(run_root)
    manifest.write_json(run_manifest_path(run_root))

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", "--run", str(run_root), "--library"])
    assert result.exit_code == 0, result.output
    assert "Library build summary" in result.output
    assert "abc123" in result.output
    assert "Library 1" in result.output


def test_summarize_library_limit_truncates(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path)

    out_file = run_root / "outputs" / "tables" / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), chunk_size=1)
    for lib_hash, lib_index in [("abc123", 1), ("def456", 2)]:
        meta = output_meta(library_hash=lib_hash, library_index=lib_index)
        rec = OutputRecord.from_sequence(
            sequence="ATGCATGCAT",
            meta=meta,
            source="densegen:demo",
            bio_type="dna",
            alphabet="dna_4",
        )
        sink.add(rec)
    sink.finalize()

    outputs_dir = run_root / "outputs" / "tables"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    attempts_df = pd.DataFrame(
        [
            {
                "attempt_id": "a1",
                "attempt_index": 1,
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:01+00:00",
                "status": "success",
                "reason": "ok",
                "detail_json": "{}",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash",
                "solution_id": "out1",
                "used_tf_counts_json": "{}",
                "used_tf_list": ["lexA", "cpxR"],
                "sampling_library_index": 1,
                "sampling_library_hash": "abc123",
                "solver_status": "optimal",
                "solver_objective": 0.0,
                "solver_solve_time_s": 0.1,
                "dense_arrays_version": None,
                "dense_arrays_version_source": "unknown",
                "library_tfbs": ["AAA", "CCC"],
                "library_tfs": ["lexA", "cpxR"],
                "library_site_ids": ["", ""],
                "library_sources": ["inputs.csv", "inputs.csv"],
            },
            {
                "attempt_id": "a2",
                "attempt_index": 2,
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:02+00:00",
                "status": "success",
                "reason": "ok",
                "detail_json": "{}",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash2",
                "solution_id": "out2",
                "used_tf_counts_json": "{}",
                "used_tf_list": ["lexA", "cpxR"],
                "sampling_library_index": 2,
                "sampling_library_hash": "def456",
                "solver_status": "optimal",
                "solver_objective": 0.0,
                "solver_solve_time_s": 0.1,
                "dense_arrays_version": None,
                "dense_arrays_version_source": "unknown",
                "library_tfbs": ["AAA", "CCC"],
                "library_tfs": ["lexA", "cpxR"],
                "library_site_ids": ["", ""],
                "library_sources": ["inputs.csv", "inputs.csv"],
            },
        ]
    )
    attempts_df.to_parquet(outputs_dir / "attempts.parquet", index=False)

    solutions_df = pd.DataFrame(
        [
            {
                "solution_id": "out1",
                "attempt_id": "a1",
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:01+00:00",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash",
                "sampling_library_index": 1,
                "sampling_library_hash": "abc123",
            },
            {
                "solution_id": "out2",
                "attempt_id": "a2",
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:02+00:00",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash2",
                "sampling_library_index": 2,
                "sampling_library_hash": "def456",
            },
        ]
    )
    solutions_df.to_parquet(outputs_dir / "solutions.parquet", index=False)

    manifest = RunManifest(
        run_id="demo",
        created_at="2026-01-14T00:00:00+00:00",
        schema_version="2.5",
        config_sha256="dummy",
        run_root=str(run_root),
        random_seed=123,
        seed_stage_a=456,
        seed_stage_b=789,
        seed_solver=101112,
        solver_backend="CBC",
        solver_strategy="iterate",
        solver_time_limit_seconds=None,
        solver_threads=None,
        solver_strands="double",
        dense_arrays_version=None,
        dense_arrays_version_source="unknown",
        items=[
            PlanManifest(
                input_name="demo_input",
                plan_name="demo_plan",
                generated=2,
                duplicates_skipped=0,
                failed_solutions=0,
                total_resamples=0,
                libraries_built=2,
                stall_events=0,
            )
        ],
    )
    ensure_run_meta_dir(run_root)
    manifest.write_json(run_manifest_path(run_root))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "inspect",
            "run",
            "--run",
            str(run_root),
            "--library",
            "--library-limit",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Showing 1 of 2 libraries" in result.output
    assert "abc123" in result.output
    assert "def456" not in result.output
