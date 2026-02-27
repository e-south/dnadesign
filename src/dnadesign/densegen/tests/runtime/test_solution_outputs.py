"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_solution_outputs.py

Tests for Stage-B solution output recording.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src.adapters.outputs.base import SinkBase
from dnadesign.densegen.src.adapters.outputs.record import OutputRecord
from dnadesign.densegen.src.core.pipeline.solution_outputs import record_solution_outputs


class _DummySink(SinkBase):
    def __init__(self) -> None:
        self.records = []

    def add(self, record: OutputRecord) -> bool:
        self.records.append(record)
        return True

    def flush(self) -> None:
        return None


def test_record_solution_outputs_emits_composition_rows(tmp_path: Path) -> None:
    sink = _DummySink()
    composition_rows: list[dict] = []
    used_tfbs_detail = [
        {
            "part_kind": "tfbs",
            "part_index": 0,
            "regulator": "TF1",
            "sequence": "AAA",
            "core_sequence": "AAA",
            "offset": 0,
            "offset_raw": 0,
            "pad_left": 0,
            "length": 3,
            "end": 3,
            "orientation": "fwd",
            "source": "unit",
            "motif_id": "motif_1",
            "tfbs_id": "tfbs_1",
        },
        {
            "part_kind": "tfbs",
            "part_index": 1,
            "regulator": "TF2",
            "sequence": "TT",
            "core_sequence": "TT",
            "offset": 3,
            "offset_raw": 3,
            "pad_left": 0,
            "length": 2,
            "end": 5,
            "orientation": "rev",
            "source": "unit",
            "motif_id": "motif_2",
            "tfbs_id": "tfbs_2",
        },
    ]
    final_seq = "AAATT"
    record_solution_outputs(
        sinks=[sink],
        final_seq=final_seq,
        derived={},
        source_label="input",
        plan_name="plan",
        output_bio_type="dna",
        output_alphabet="dna",
        tables_root=tmp_path,
        run_id="run",
        next_attempt_index=lambda: 1,
        used_tf_counts={},
        used_tf_list=[],
        sampling_library_index=1,
        sampling_library_hash="hash",
        solver_status=None,
        solver_objective=None,
        solver_solve_time_s=None,
        dense_arrays_version=None,
        dense_arrays_version_source="package",
        library_tfbs=[],
        library_tfs=[],
        library_site_ids=[],
        library_sources=[],
        attempts_buffer=[],
        solution_rows=None,
        composition_rows=composition_rows,
        events_path=None,
        used_tfbs=[],
        used_tfbs_detail=used_tfbs_detail,
    )

    expected_id = OutputRecord.from_sequence(
        sequence=final_seq,
        meta={},
        source="input",
        bio_type="dna",
        alphabet="dna",
    ).id
    assert len(composition_rows) == len(used_tfbs_detail)
    assert {row["solution_id"] for row in composition_rows} == {expected_id}
    assert {row["regulator"] for row in composition_rows} == {"TF1", "TF2"}


def test_record_solution_outputs_preserves_stage_a_lineage_in_composition_rows(tmp_path: Path) -> None:
    sink = _DummySink()
    composition_rows: list[dict] = []
    used_tfbs_detail = [
        {
            "part_kind": "tfbs",
            "part_index": 0,
            "regulator": "lexA",
            "sequence": "AAA",
            "core_sequence": "AAA",
            "orientation": "fwd",
            "offset": 0,
            "offset_raw": 0,
            "pad_left": 0,
            "length": 3,
            "end": 3,
            "source": "unit",
            "motif_id": "motif_1",
            "tfbs_id": "tfbs_1",
            "score_best_hit_raw": 8.25,
            "score_theoretical_max": 9.50,
            "score_relative_to_theoretical_max": 0.87,
            "rank_among_mined_positive": 2,
            "rank_among_selected": 3,
            "selection_policy": "mmr",
            "nearest_selected_similarity": 0.12,
            "nearest_selected_distance": 4.0,
            "nearest_selected_distance_norm": 0.50,
            "matched_start": 14,
            "matched_stop": 17,
            "matched_strand": "+",
        }
    ]
    record_solution_outputs(
        sinks=[sink],
        final_seq="AAATT",
        derived={},
        source_label="input",
        plan_name="plan",
        output_bio_type="dna",
        output_alphabet="dna",
        tables_root=tmp_path,
        run_id="run",
        next_attempt_index=lambda: 1,
        used_tf_counts={},
        used_tf_list=[],
        sampling_library_index=1,
        sampling_library_hash="hash",
        solver_status=None,
        solver_objective=None,
        solver_solve_time_s=None,
        dense_arrays_version=None,
        dense_arrays_version_source="package",
        library_tfbs=[],
        library_tfs=[],
        library_site_ids=[],
        library_sources=[],
        attempts_buffer=[],
        solution_rows=None,
        composition_rows=composition_rows,
        events_path=None,
        used_tfbs=[],
        used_tfbs_detail=used_tfbs_detail,
    )

    assert len(composition_rows) == 1
    row = composition_rows[0]
    assert row["score_best_hit_raw"] == 8.25
    assert row["score_theoretical_max"] == 9.50
    assert row["score_relative_to_theoretical_max"] == 0.87
    assert row["rank_among_mined_positive"] == 2
    assert row["rank_among_selected"] == 3
    assert row["selection_policy"] == "mmr"
    assert row["nearest_selected_similarity"] == 0.12
    assert row["nearest_selected_distance"] == 4.0
    assert row["nearest_selected_distance_norm"] == 0.50
    assert row["matched_start"] == 14
    assert row["matched_stop"] == 17
    assert row["matched_strand"] == "+"


def test_record_solution_outputs_records_structured_solution_rows(tmp_path: Path) -> None:
    sink = _DummySink()
    attempts_buffer: list[dict] = []
    solution_rows: list[dict] = []
    calls = {"n": 0}

    def _next_attempt_index() -> int:
        calls["n"] += 1
        return calls["n"]

    final_seq = "AACCGGTT"
    accepted = record_solution_outputs(
        sinks=[sink],
        final_seq=final_seq,
        derived={"created_at": "2026-02-06T00:00:00+00:00"},
        source_label="plan_pool__demo",
        plan_name="demo_plan",
        output_bio_type="dna",
        output_alphabet="dna",
        tables_root=tmp_path,
        run_id="run_demo",
        next_attempt_index=_next_attempt_index,
        used_tf_counts={"TF_A": 1},
        used_tf_list=["TF_A"],
        sampling_library_index=7,
        sampling_library_hash="library_hash_123",
        solver_status="optimal",
        solver_objective=1.0,
        solver_solve_time_s=0.5,
        dense_arrays_version="0.1.0",
        dense_arrays_version_source="installed",
        library_tfbs=["AAA"],
        library_tfs=["TF_A"],
        library_site_ids=["site_1"],
        library_sources=["input.csv"],
        attempts_buffer=attempts_buffer,
        solution_rows=solution_rows,
        composition_rows=None,
        events_path=None,
        used_tfbs=["AAA"],
        used_tfbs_detail=[],
    )

    assert accepted is True
    assert calls["n"] == 1
    assert len(attempts_buffer) == 1
    assert len(solution_rows) == 1

    attempt = attempts_buffer[0]
    row = solution_rows[0]
    assert row["attempt_id"] == attempt["attempt_id"]
    assert row["run_id"] == "run_demo"
    assert row["input_name"] == "plan_pool__demo"
    assert row["plan_name"] == "demo_plan"
    assert row["sequence"] == final_seq
    assert row["sampling_library_index"] == 7
    assert row["sampling_library_hash"] == "library_hash_123"
