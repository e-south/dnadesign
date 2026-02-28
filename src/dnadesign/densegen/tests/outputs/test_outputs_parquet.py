"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_outputs_parquet.py

Parquet output metadata contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.densegen.src.adapters.outputs import OutputRecord, ParquetSink


def _dummy_meta() -> dict:
    return {
        "schema_version": "2.9",
        "run_id": "demo",
        "created_at": "2026-01-14T00:00:00+00:00",
        "length": 4,
        "plan": "default",
        "input_name": "demo",
        "input_mode": "binding_sites",
        "input_pwm_ids": [],
        "used_tfbs": [],
        "used_tfbs_detail": [],
        "used_tf_counts": [],
        "library_unique_tf_count": 0,
        "library_unique_tfbs_count": 0,
        "covers_all_tfs_in_solution": True,
        "required_regulators": [],
        "min_count_by_regulator": [],
        "compression_ratio": None,
        "sampling_library_hash": "library_hash",
        "sampling_library_index": 1,
        "sequence_validation": {"validation_passed": True, "violations": []},
        "pad_used": False,
        "pad_bases": None,
        "pad_end": None,
        "pad_literal": None,
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


def test_parquet_sink_accepts_pwm_id_list(tmp_path: Path) -> None:
    out_file = tmp_path / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), chunk_size=1)
    meta = _dummy_meta()
    meta["input_mode"] = "pwm_sampled"
    meta["input_pwm_ids"] = ["motif_A", "motif_B"]
    rec = OutputRecord.from_sequence(
        sequence="ATGC",
        meta=meta,
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
