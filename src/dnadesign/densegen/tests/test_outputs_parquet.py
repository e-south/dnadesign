from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.densegen.src.adapters.outputs import OutputRecord, ParquetSink


def _dummy_meta() -> dict:
    return {
        "schema_version": "2.4",
        "run_id": "demo",
        "run_root": ".",
        "run_config_path": "config.yaml",
        "run_config_sha256": "dummy",
        "created_at": "2026-01-14T00:00:00+00:00",
        "length": 4,
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
        "plan": "default",
        "tf_list": [],
        "tfbs_parts": [],
        "used_tfbs": [],
        "used_tfbs_detail": [],
        "used_tf_counts": [],
        "used_tf_list": [],
        "covers_all_tfs_in_solution": True,
        "min_count_per_tf": 0,
        "input_type": "binding_sites",
        "input_name": "demo",
        "input_path": "inputs.csv",
        "input_dataset": None,
        "input_root": None,
        "input_mode": "binding_sites",
        "input_pwm_ids": [],
        "input_row_count": 0,
        "input_tf_count": 0,
        "input_tfbs_count": 0,
        "input_tf_tfbs_pair_count": 1,
        "sampling_fraction": None,
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
        "library_size": 0,
        "library_unique_tf_count": 0,
        "library_unique_tfbs_count": 0,
        "sequence_length": 4,
        "promoter_constraint": None,
        "sampling_target_length": 0,
        "sampling_achieved_length": 0,
        "sampling_relaxed_cap": False,
        "sampling_final_cap": None,
        "sampling_pool_strategy": "subsample",
        "sampling_library_size": 0,
        "sampling_library_strategy": None,
        "sampling_iterative_max_libraries": 0,
        "sampling_iterative_min_new_solutions": 0,
        "sampling_library_index": 1,
        "sampling_library_hash": "dummy",
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


def _count_rows(path: Path) -> int:
    table = pq.read_table(path)
    return table.num_rows


def test_parquet_sink_writes_file(tmp_path: Path) -> None:
    out_file = tmp_path / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), chunk_size=2)
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta=_dummy_meta(),
        source="src",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert sink.add(rec) is True
    sink.finalize()
    assert _count_rows(out_file) == 1


def test_parquet_sink_deduplicates(tmp_path: Path) -> None:
    out_file = tmp_path / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), deduplicate=True, chunk_size=1)
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta=_dummy_meta(),
        source="src",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert sink.add(rec) is True
    assert sink.add(rec) is False
    sink.finalize()
    assert _count_rows(out_file) == 1


def test_parquet_index_created(tmp_path: Path) -> None:
    out_file = tmp_path / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), deduplicate=True, chunk_size=1)
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta=_dummy_meta(),
        source="src",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert sink.add(rec) is True
    sink.finalize()
    assert (out_file.parent / "_densegen_ids.sqlite").exists()
    digest = sink.alignment_digest()
    assert digest is not None
    assert digest.id_count == 1


def test_parquet_sink_rejects_directory_path(tmp_path: Path) -> None:
    dir_path = tmp_path / "out_dir"
    dir_path.mkdir(parents=True, exist_ok=True)
    try:
        ParquetSink(path=str(dir_path))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for directory path")


def test_parquet_sink_rejects_mismatched_bio_type(tmp_path: Path) -> None:
    out_dir = tmp_path / "pq"
    sink = ParquetSink(path=str(out_dir), bio_type="dna", alphabet="dna_4")
    rec = OutputRecord.from_sequence(
        sequence="AUGC",
        meta={},
        source="src",
        bio_type="rna",
        alphabet="rna_4",
    )
    try:
        sink.add(rec)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for bio_type/alphabet mismatch")


def test_parquet_schema_mismatch_rejected(tmp_path: Path) -> None:
    out_file = tmp_path / "dense_arrays.parquet"
    table = pa.table(
        {
            "id": ["x"],
            "sequence": ["ATGC"],
            "bio_type": ["dna"],
            "alphabet": ["dna_4"],
            "source": ["src"],
        }
    )
    pq.write_table(table, out_file)
    with pytest.raises(RuntimeError, match="schema"):
        ParquetSink(path=str(out_file))


def test_parquet_existing_ids_loaded_without_dedup(tmp_path: Path) -> None:
    out_file = tmp_path / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), deduplicate=True, chunk_size=1)
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta=_dummy_meta(),
        source="src",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert sink.add(rec) is True
    sink.finalize()

    sink2 = ParquetSink(path=str(out_file), deduplicate=False, chunk_size=1)
    existing = sink2.existing_ids()
    assert rec.id in existing
