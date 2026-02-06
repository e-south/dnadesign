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
        {"tf": "TF1", "tfbs": "AAA", "offset": 0, "length": 3, "end": 3, "orientation": "fwd"},
        {"tf": "TF2", "tfbs": "TT", "offset": 3, "length": 2, "end": 5, "orientation": "rev"},
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
    assert {row["tf"] for row in composition_rows} == {"TF1", "TF2"}


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
