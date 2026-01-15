from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from dnadesign.densegen.src.adapters.outputs import OutputRecord, ParquetSink


def _dummy_meta() -> dict:
    return {
        "schema_version": "2.1",
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
        "input_pwm_strategy": None,
        "input_pwm_score_threshold": None,
        "input_pwm_score_percentile": None,
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
        "sampling_iterative_max_libraries": 0,
        "sampling_iterative_min_new_solutions": 0,
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
    }


def _count_rows(path: Path) -> int:
    dataset = ds.dataset(path, format="parquet")
    return dataset.count_rows()


def test_parquet_sink_writes_dataset(tmp_path: Path) -> None:
    out_dir = tmp_path / "pq"
    sink = ParquetSink(path=str(out_dir), chunk_size=2)
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta=_dummy_meta(),
        source="src",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert sink.add(rec) is True
    sink.flush()
    assert _count_rows(out_dir) == 1


def test_parquet_sink_deduplicates(tmp_path: Path) -> None:
    out_dir = tmp_path / "pq"
    sink = ParquetSink(path=str(out_dir), deduplicate=True, chunk_size=1)
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta=_dummy_meta(),
        source="src",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert sink.add(rec) is True
    assert sink.add(rec) is False
    sink.flush()
    assert _count_rows(out_dir) == 1


def test_parquet_index_created(tmp_path: Path) -> None:
    out_dir = tmp_path / "pq"
    sink = ParquetSink(path=str(out_dir), deduplicate=True, chunk_size=1)
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta=_dummy_meta(),
        source="src",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert sink.add(rec) is True
    sink.flush()
    assert (out_dir / "_densegen_ids.sqlite").exists()
    digest = sink.alignment_digest()
    assert digest is not None
    assert digest.id_count == 1


def test_parquet_sink_rejects_file_path(tmp_path: Path) -> None:
    file_path = tmp_path / "out.parquet"
    file_path.write_text("not parquet")
    try:
        ParquetSink(path=str(file_path))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for file path")


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
    out_dir = tmp_path / "pq"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": ["x"],
            "sequence": ["ATGC"],
            "bio_type": ["dna"],
            "alphabet": ["dna_4"],
            "source": ["src"],
        }
    )
    pq.write_table(table, out_dir / "part-000.parquet")
    with pytest.raises(RuntimeError, match="schema"):
        ParquetSink(path=str(out_dir))


def test_parquet_existing_ids_loaded_without_dedup(tmp_path: Path) -> None:
    out_dir = tmp_path / "pq"
    sink = ParquetSink(path=str(out_dir), deduplicate=True, chunk_size=1)
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta=_dummy_meta(),
        source="src",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert sink.add(rec) is True
    sink.flush()

    sink2 = ParquetSink(path=str(out_dir), deduplicate=False, chunk_size=1)
    existing = sink2.existing_ids()
    assert rec.id in existing
