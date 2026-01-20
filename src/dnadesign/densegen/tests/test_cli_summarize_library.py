from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.densegen.src.adapters.outputs import OutputRecord, ParquetSink
from dnadesign.densegen.src.cli import app
from dnadesign.densegen.src.core.run_manifest import PlanManifest, RunManifest
from dnadesign.densegen.src.core.run_paths import ensure_run_meta_dir, run_manifest_path


def _base_meta(library_hash: str, library_index: int) -> dict:
    return {
        "schema_version": "2.4",
        "run_id": "demo",
        "run_root": ".",
        "run_config_path": "config.yaml",
        "run_config_sha256": "dummy",
        "created_at": "2026-01-14T00:00:00+00:00",
        "length": 10,
        "random_seed": 0,
        "policy_gc_fill": "off",
        "policy_sampling": "subsample",
        "policy_solver": "iterate",
        "solver_backend": "CBC",
        "solver_strategy": "iterate",
        "solver_options": [],
        "solver_strands": "double",
        "dense_arrays_version": None,
        "dense_arrays_version_source": "unknown",
        "solver_status": None,
        "solver_objective": None,
        "solver_solve_time_s": None,
        "plan": "demo_plan",
        "tf_list": ["lexA", "cpxR"],
        "tfbs_parts": ["lexA:AAA", "cpxR:CCC"],
        "used_tfbs": ["lexA:AAA", "cpxR:CCC"],
        "used_tfbs_detail": [
            {"tf": "lexA", "tfbs": "AAA", "orientation": "fwd", "offset": 0},
            {"tf": "cpxR", "tfbs": "CCC", "orientation": "fwd", "offset": 4},
        ],
        "used_tf_counts": [{"tf": "lexA", "count": 1}, {"tf": "cpxR", "count": 1}],
        "used_tf_list": ["lexA", "cpxR"],
        "covers_all_tfs_in_solution": True,
        "min_count_per_tf": 0,
        "input_type": "binding_sites",
        "input_name": "demo_input",
        "input_path": "inputs.csv",
        "input_dataset": None,
        "input_root": None,
        "input_mode": "binding_sites",
        "input_pwm_ids": [],
        "input_row_count": 0,
        "input_tf_count": 0,
        "input_tfbs_count": 0,
        "input_tf_tfbs_pair_count": 1,
        "sampling_fraction": 0.5,
        "sampling_fraction_pairs": 0.5,
        "input_pwm_strategy": None,
        "input_pwm_scoring_backend": None,
        "input_pwm_score_threshold": None,
        "input_pwm_score_percentile": None,
        "input_pwm_pvalue_threshold": None,
        "input_pwm_pvalue_bins": None,
        "input_pwm_mining_batch_size": None,
        "input_pwm_mining_max_batches": None,
        "input_pwm_mining_max_candidates": None,
        "input_pwm_mining_max_seconds": None,
        "input_pwm_mining_retain_bin_ids": None,
        "input_pwm_mining_log_every_batches": None,
        "input_pwm_selection_policy": None,
        "input_pwm_bgfile": None,
        "input_pwm_keep_all_candidates_debug": None,
        "input_pwm_include_matched_sequence": None,
        "input_pwm_n_sites": None,
        "input_pwm_oversample_factor": None,
        "fixed_elements": {"promoter_constraints": [], "side_biases": {"left": [], "right": []}},
        "visual": "",
        "compression_ratio": None,
        "library_size": 2,
        "library_unique_tf_count": 2,
        "library_unique_tfbs_count": 2,
        "sequence_length": 10,
        "promoter_constraint": None,
        "sampling_target_length": 0,
        "sampling_achieved_length": 0,
        "sampling_relaxed_cap": False,
        "sampling_final_cap": None,
        "sampling_pool_strategy": "subsample",
        "sampling_library_size": 2,
        "sampling_library_strategy": "tf_balanced",
        "sampling_iterative_max_libraries": 1,
        "sampling_iterative_min_new_solutions": 0,
        "sampling_library_index": library_index,
        "sampling_library_hash": library_hash,
        "required_regulators": [],
        "min_required_regulators": None,
        "min_count_by_regulator": [],
        "covers_required_regulators": True,
        "gap_fill_used": False,
        "gap_fill_bases": None,
        "gap_fill_end": None,
        "gap_fill_gc_min": None,
        "gap_fill_gc_max": None,
        "gap_fill_gc_target_min": None,
        "gap_fill_gc_target_max": None,
        "gap_fill_gc_actual": None,
        "gap_fill_relaxed": None,
        "gap_fill_attempts": None,
        "gc_total": 0.5,
        "gc_core": 0.5,
    }


def _write_config(path: Path) -> None:
    path.write_text(
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


def test_summarize_library_grouping(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path)

    # outputs
    out_file = run_root / "outputs" / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), chunk_size=1)
    meta = _base_meta(library_hash="abc123", library_index=1)
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
    outputs_dir = run_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    attempts_df = pd.DataFrame(
        [
            {
                "attempt_id": "a1",
                "run_id": "demo",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:01+00:00",
                "status": "success",
                "reason": "ok",
                "detail_json": "{}",
                "sequence": "ATGCATGCAT",
                "sequence_hash": "hash",
                "output_id": "out1",
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

    # run manifest
    manifest = RunManifest(
        run_id="demo",
        created_at="2026-01-14T00:00:00+00:00",
        schema_version="2.4",
        config_sha256="dummy",
        run_root=str(run_root),
        random_seed=123,
        seed_stage_a=456,
        seed_stage_b=789,
        seed_solver=101112,
        solver_backend="CBC",
        solver_strategy="iterate",
        solver_options=[],
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
