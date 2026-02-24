"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_stage_b_solution_flow.py

Unit tests for Stage-B solution flow helper utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dnadesign.densegen.src.core.pipeline.stage_b_solution_persistence import (
    persist_candidate_solution,
)
from dnadesign.densegen.src.core.pipeline.stage_b_solution_rejections import (
    emit_sequence_validation_failed_event,
    enforce_failed_solution_cap,
    handle_duplicate_sequence,
    mark_stall_detected,
    reject_sequence_validation_failure,
    reject_solution_and_enforce_cap,
    reject_solution_requirement_failure,
    resolve_failed_solution_cap,
)
from dnadesign.densegen.src.core.pipeline.stage_b_solution_types import StageBRejectionContext


def test_enforce_failed_solution_cap_allows_within_limit() -> None:
    enforce_failed_solution_cap(
        failed_solutions=2,
        max_failed_solutions=2,
        source_label="demo",
        plan_name="baseline",
    )


def test_enforce_failed_solution_cap_raises_with_cause() -> None:
    cause = ValueError("boom")
    with pytest.raises(RuntimeError, match="Exceeded max_failed_solutions=1") as exc:
        enforce_failed_solution_cap(
            failed_solutions=2,
            max_failed_solutions=1,
            source_label="demo",
            plan_name="baseline",
            cause=cause,
        )
    assert exc.value.__cause__ is cause


def test_reject_solution_and_enforce_cap_records_rejection_and_returns_count(tmp_path) -> None:
    attempts_buffer: list[dict] = []
    next_index = 0

    def _next_attempt_index() -> int:
        nonlocal next_index
        next_index += 1
        return next_index

    context = StageBRejectionContext(
        source_label="demo",
        plan_name="baseline",
        tables_root=tmp_path,
        run_id="run-1",
        dense_arrays_version="1.0.0",
        dense_arrays_version_source="metadata",
        attempts_buffer=attempts_buffer,
        next_attempt_index=_next_attempt_index,
    )
    failed_solutions = reject_solution_and_enforce_cap(
        context=context,
        reason="required_regulators",
        detail={"required_regulators": ["lexA"]},
        sequence="ACGT",
        used_tf_counts={"lexA": 0},
        used_tf_list=[],
        sampling_library_index=3,
        sampling_library_hash="abc123",
        solver_status="optimal",
        solver_objective=1.0,
        solver_solve_time_s=0.1,
        library_tfbs=["site-1"],
        library_tfs=["lexA"],
        library_site_ids=["site-1"],
        library_sources=["sampling"],
        failed_solutions=0,
        max_failed_solutions=5,
        source_label="demo",
        plan_name="baseline",
    )
    assert failed_solutions == 1
    assert len(attempts_buffer) == 1
    rejection = attempts_buffer[0]
    assert rejection["status"] == "rejected"
    assert rejection["reason"] == "required_regulators"
    assert rejection["attempt_index"] == 1


def test_reject_solution_and_enforce_cap_raises_when_cap_exceeded(tmp_path) -> None:
    attempts_buffer: list[dict] = []
    next_index = 0

    def _next_attempt_index() -> int:
        nonlocal next_index
        next_index += 1
        return next_index

    context = StageBRejectionContext(
        source_label="demo",
        plan_name="baseline",
        tables_root=tmp_path,
        run_id="run-1",
        dense_arrays_version="1.0.0",
        dense_arrays_version_source="metadata",
        attempts_buffer=attempts_buffer,
        next_attempt_index=_next_attempt_index,
    )
    cause = ValueError("validation failed")
    with pytest.raises(RuntimeError, match="Exceeded max_failed_solutions=1") as exc:
        reject_solution_and_enforce_cap(
            context=context,
            reason="sequence_validation_failed",
            detail={"violations": [{"pattern": "TTGACA"}]},
            sequence="TTGACA",
            used_tf_counts={"lexA": 1},
            used_tf_list=["lexA"],
            sampling_library_index=3,
            sampling_library_hash="abc123",
            solver_status="optimal",
            solver_objective=1.0,
            solver_solve_time_s=0.1,
            library_tfbs=["site-1"],
            library_tfs=["lexA"],
            library_site_ids=["site-1"],
            library_sources=["sampling"],
            failed_solutions=1,
            max_failed_solutions=1,
            source_label="demo",
            plan_name="baseline",
            cause=cause,
        )
    assert exc.value.__cause__ is cause
    assert len(attempts_buffer) == 1


def test_emit_sequence_validation_failed_event_noops_without_events_path() -> None:
    logger = MagicMock()
    emit_event = MagicMock()
    emit_sequence_validation_failed_event(
        events_path=None,
        payload={"reason": "bad-seq"},
        emit_event=emit_event,
        logger=logger,
    )
    emit_event.assert_not_called()
    logger.debug.assert_not_called()


def test_emit_sequence_validation_failed_event_logs_debug_on_emit_failure(tmp_path: Path) -> None:
    logger = MagicMock()

    def _emit_event(*_args, **_kwargs) -> None:
        raise ValueError("boom")

    emit_sequence_validation_failed_event(
        events_path=tmp_path / "events.jsonl",
        payload={"reason": "bad-seq"},
        emit_event=_emit_event,
        logger=logger,
    )
    logger.debug.assert_called_once()


def test_mark_stall_detected_updates_state_and_emits_event(tmp_path: Path) -> None:
    events: list[tuple[Path, str, dict]] = []
    logger = MagicMock()

    def _emit_event(path: Path, *, event: str, payload: dict) -> None:
        events.append((path, event, payload))

    stall_events, stall_triggered = mark_stall_detected(
        events_path=tmp_path / "events.jsonl",
        source_label="demo",
        plan_name="baseline",
        stall_seconds=30,
        last_progress=10.0,
        now=41.5,
        sampling_library_index=2,
        sampling_library_hash="abc123",
        stall_events=1,
        stall_triggered=False,
        emit_event=_emit_event,
        logger=logger,
    )
    assert stall_events == 2
    assert stall_triggered is True
    logger.info.assert_called_once()
    assert events == [
        (
            tmp_path / "events.jsonl",
            "STALL_DETECTED",
            {
                "input_name": "demo",
                "plan_name": "baseline",
                "stall_seconds": 31.5,
                "library_index": 2,
                "library_hash": "abc123",
            },
        )
    ]


def test_mark_stall_detected_is_noop_when_already_triggered(tmp_path: Path) -> None:
    logger = MagicMock()
    emit_event = MagicMock()
    stall_events, stall_triggered = mark_stall_detected(
        events_path=tmp_path / "events.jsonl",
        source_label="demo",
        plan_name="baseline",
        stall_seconds=30,
        last_progress=10.0,
        now=41.5,
        sampling_library_index=2,
        sampling_library_hash="abc123",
        stall_events=7,
        stall_triggered=True,
        emit_event=emit_event,
        logger=logger,
    )
    assert stall_events == 7
    assert stall_triggered is True
    logger.info.assert_not_called()
    emit_event.assert_not_called()


def test_mark_stall_detected_raises_if_event_emit_fails(tmp_path: Path) -> None:
    logger = MagicMock()

    def _emit_event(*_args, **_kwargs) -> None:
        raise RuntimeError("emit failed")

    with pytest.raises(RuntimeError, match="Failed to emit STALL_DETECTED event.") as exc:
        mark_stall_detected(
            events_path=tmp_path / "events.jsonl",
            source_label="demo",
            plan_name="baseline",
            stall_seconds=30,
            last_progress=10.0,
            now=41.5,
            sampling_library_index=2,
            sampling_library_hash="abc123",
            stall_events=0,
            stall_triggered=False,
            emit_event=_emit_event,
            logger=logger,
        )
    assert exc.value.__cause__ is not None


def test_reject_sequence_validation_failure_emits_event_and_returns_count(tmp_path: Path) -> None:
    attempts_buffer: list[dict] = []
    next_index = 0
    events: list[tuple[Path, str, dict]] = []
    logger = MagicMock()

    def _next_attempt_index() -> int:
        nonlocal next_index
        next_index += 1
        return next_index

    def _emit_event(path: Path, *, event: str, payload: dict) -> None:
        events.append((path, event, payload))

    context = StageBRejectionContext(
        source_label="demo",
        plan_name="baseline",
        tables_root=tmp_path,
        run_id="run-1",
        dense_arrays_version="1.0.0",
        dense_arrays_version_source="metadata",
        attempts_buffer=attempts_buffer,
        next_attempt_index=_next_attempt_index,
    )
    failed_solutions = reject_sequence_validation_failure(
        rejection_context=context,
        rejection_detail={"violations": [{"pattern": "TTGACA"}]},
        rejection_event_payload={"violations": [{"pattern": "TTGACA"}]},
        validation_error=None,
        final_seq="TTGACA",
        used_tf_counts={"lexA": 1},
        used_tf_list=["lexA"],
        sampling_library_index=2,
        sampling_library_hash="abc123",
        solver_status="optimal",
        solver_objective=1.0,
        solver_solve_time_s=0.1,
        library_tfbs=["site-1"],
        library_tfs=["lexA"],
        library_site_ids=["site-1"],
        library_sources=["sampling"],
        failed_solutions=0,
        max_failed_solutions=10,
        source_label="demo",
        plan_name="baseline",
        events_path=tmp_path / "events.jsonl",
        emit_event=_emit_event,
        logger=logger,
    )
    assert failed_solutions == 1
    assert len(attempts_buffer) == 1
    assert attempts_buffer[0]["reason"] == "sequence_validation_failed"
    assert events == [
        (
            tmp_path / "events.jsonl",
            "SEQUENCE_VALIDATION_FAILED",
            {"violations": [{"pattern": "TTGACA"}]},
        )
    ]


def test_reject_sequence_validation_failure_allows_retries_when_cap_is_zero(tmp_path: Path) -> None:
    attempts_buffer: list[dict] = []
    next_index = 0
    logger = MagicMock()

    def _next_attempt_index() -> int:
        nonlocal next_index
        next_index += 1
        return next_index

    context = StageBRejectionContext(
        source_label="demo",
        plan_name="baseline",
        tables_root=tmp_path,
        run_id="run-1",
        dense_arrays_version="1.0.0",
        dense_arrays_version_source="metadata",
        attempts_buffer=attempts_buffer,
        next_attempt_index=_next_attempt_index,
    )
    cause = ValueError("invalid placement")
    failed_solutions = reject_sequence_validation_failure(
        rejection_context=context,
        rejection_detail={"error": "invalid placement"},
        rejection_event_payload=None,
        validation_error=cause,
        final_seq="TTGACA",
        used_tf_counts={"lexA": 1},
        used_tf_list=["lexA"],
        sampling_library_index=2,
        sampling_library_hash="abc123",
        solver_status="optimal",
        solver_objective=1.0,
        solver_solve_time_s=0.1,
        library_tfbs=["site-1"],
        library_tfs=["lexA"],
        library_site_ids=["site-1"],
        library_sources=["sampling"],
        failed_solutions=0,
        max_failed_solutions=0,
        source_label="demo",
        plan_name="baseline",
        events_path=tmp_path / "events.jsonl",
        emit_event=MagicMock(),
        logger=logger,
    )
    assert failed_solutions == 1
    assert len(attempts_buffer) == 1


def test_resolve_failed_solution_cap_uses_scaled_and_absolute_limits() -> None:
    assert resolve_failed_solution_cap(max_failed_solutions=0, max_failed_solutions_per_target=0.0, quota=100) == 0
    assert resolve_failed_solution_cap(max_failed_solutions=0, max_failed_solutions_per_target=0.2, quota=100) == 20
    assert resolve_failed_solution_cap(max_failed_solutions=10, max_failed_solutions_per_target=0.0, quota=100) == 10
    assert resolve_failed_solution_cap(max_failed_solutions=50, max_failed_solutions_per_target=0.2, quota=100) == 20


def test_handle_duplicate_sequence_accepts_new_sequence() -> None:
    fingerprints = {"AAAA"}
    logger = MagicMock()
    should_continue, should_break, duplicate_solutions, consecutive_dup = handle_duplicate_sequence(
        sequence="CCCC",
        fingerprints=fingerprints,
        duplicate_solutions=2,
        consecutive_dup=1,
        max_dupes=3,
        source_label="demo",
        plan_name="baseline",
        logger=logger,
    )
    assert should_continue is False
    assert should_break is False
    assert duplicate_solutions == 2
    assert consecutive_dup == 0
    assert "CCCC" in fingerprints
    logger.info.assert_not_called()


def test_handle_duplicate_sequence_continues_when_below_cap() -> None:
    fingerprints = {"AAAA"}
    logger = MagicMock()
    should_continue, should_break, duplicate_solutions, consecutive_dup = handle_duplicate_sequence(
        sequence="AAAA",
        fingerprints=fingerprints,
        duplicate_solutions=2,
        consecutive_dup=1,
        max_dupes=3,
        source_label="demo",
        plan_name="baseline",
        logger=logger,
    )
    assert should_continue is True
    assert should_break is False
    assert duplicate_solutions == 3
    assert consecutive_dup == 2
    logger.info.assert_not_called()


def test_handle_duplicate_sequence_breaks_when_cap_reached() -> None:
    fingerprints = {"AAAA"}
    logger = MagicMock()
    should_continue, should_break, duplicate_solutions, consecutive_dup = handle_duplicate_sequence(
        sequence="AAAA",
        fingerprints=fingerprints,
        duplicate_solutions=2,
        consecutive_dup=2,
        max_dupes=3,
        source_label="demo",
        plan_name="baseline",
        logger=logger,
    )
    assert should_continue is False
    assert should_break is True
    assert duplicate_solutions == 3
    assert consecutive_dup == 3
    logger.info.assert_called_once()


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("min_count_per_tf", (1, 0, 0)),
        ("required_regulators", (0, 1, 0)),
        ("min_count_by_regulator", (0, 0, 1)),
    ],
)
def test_reject_solution_requirement_failure_updates_specific_counter(
    tmp_path: Path,
    reason: str,
    expected: tuple[int, int, int],
) -> None:
    attempts_buffer: list[dict] = []
    next_index = 0
    diagnostics = MagicMock()

    def _next_attempt_index() -> int:
        nonlocal next_index
        next_index += 1
        return next_index

    context = StageBRejectionContext(
        source_label="demo",
        plan_name="baseline",
        tables_root=tmp_path,
        run_id="run-1",
        dense_arrays_version="1.0.0",
        dense_arrays_version_source="metadata",
        attempts_buffer=attempts_buffer,
        next_attempt_index=_next_attempt_index,
    )
    failed_solutions, min_count_per_tf_count, required_regulators_count, min_count_by_regulator_count = (
        reject_solution_requirement_failure(
            rejection_context=context,
            rejection_reason=reason,
            rejection_detail={reason: True},
            sequence="ACGT",
            used_tf_counts={"lexA": 0},
            used_tf_list=[],
            sampling_library_index=2,
            sampling_library_hash="abc123",
            solver_status="optimal",
            solver_objective=1.0,
            solver_solve_time_s=0.1,
            library_tfbs=["site-1"],
            library_tfs=["lexA"],
            library_site_ids=["site-1"],
            library_sources=["sampling"],
            failed_solutions=0,
            max_failed_solutions=10,
            source_label="demo",
            plan_name="baseline",
            failed_min_count_per_tf=0,
            failed_required_regulators=0,
            failed_min_count_by_regulator=0,
            diagnostics=diagnostics,
        )
    )
    assert failed_solutions == 1
    assert (min_count_per_tf_count, required_regulators_count, min_count_by_regulator_count) == expected
    diagnostics.record_site_failures.assert_called_once_with(reason)
    assert len(attempts_buffer) == 1
    assert attempts_buffer[0]["reason"] == reason


def test_reject_solution_requirement_failure_rejects_unknown_reason(tmp_path: Path) -> None:
    attempts_buffer: list[dict] = []
    next_index = 0
    diagnostics = MagicMock()

    def _next_attempt_index() -> int:
        nonlocal next_index
        next_index += 1
        return next_index

    context = StageBRejectionContext(
        source_label="demo",
        plan_name="baseline",
        tables_root=tmp_path,
        run_id="run-1",
        dense_arrays_version="1.0.0",
        dense_arrays_version_source="metadata",
        attempts_buffer=attempts_buffer,
        next_attempt_index=_next_attempt_index,
    )
    with pytest.raises(RuntimeError, match="Unsupported requirement rejection reason"):
        reject_solution_requirement_failure(
            rejection_context=context,
            rejection_reason="unknown_rule",
            rejection_detail={"unknown_rule": True},
            sequence="ACGT",
            used_tf_counts={"lexA": 0},
            used_tf_list=[],
            sampling_library_index=2,
            sampling_library_hash="abc123",
            solver_status="optimal",
            solver_objective=1.0,
            solver_solve_time_s=0.1,
            library_tfbs=["site-1"],
            library_tfs=["lexA"],
            library_site_ids=["site-1"],
            library_sources=["sampling"],
            failed_solutions=0,
            max_failed_solutions=10,
            source_label="demo",
            plan_name="baseline",
            failed_min_count_per_tf=0,
            failed_required_regulators=0,
            failed_min_count_by_regulator=0,
            diagnostics=diagnostics,
        )
    diagnostics.record_site_failures.assert_not_called()
    assert attempts_buffer == []


def test_persist_candidate_solution_skips_when_metadata_missing() -> None:
    def _write_output(**_kwargs):
        return "skipped_no_metadata", []

    def _record_progress(**_kwargs):
        raise AssertionError("progress should not be recorded for skipped metadata")

    global_generated, local_generated, produced_this_library, duplicate_records, accepted = persist_candidate_solution(
        solution_output_context=object(),
        progress_context=object(),
        sol=object(),
        seq="ACGT",
        final_seq="ACGT",
        used_tfbs=[],
        used_tfbs_detail=[],
        used_tf_counts={},
        used_tf_list=[],
        pad_meta={},
        covers_all=True,
        covers_required=True,
        tfbs_parts=[],
        regulator_labels=[],
        library_for_opt=[],
        sampling_info={},
        required_regulators=[],
        min_required_regulators=None,
        sampling_fraction=None,
        sampling_fraction_pairs=None,
        sampling_library_index=1,
        sampling_library_hash="h1",
        library_tfbs=[],
        library_tfs=[],
        library_site_ids=[],
        library_sources=[],
        promoter_detail={},
        sequence_validation={},
        solver_status="optimal",
        solver_objective=1.0,
        solver_solve_time_s=0.1,
        apply_pad_offsets=lambda detail, _pad: detail,
        global_generated=2,
        local_generated=1,
        produced_this_library=1,
        duplicate_records=0,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        write_output_fn=_write_output,
        record_progress_fn=_record_progress,
    )
    assert accepted is False
    assert (global_generated, local_generated, produced_this_library, duplicate_records) == (2, 1, 1, 0)


def test_persist_candidate_solution_counts_duplicate_records() -> None:
    def _write_output(**_kwargs):
        return "duplicate", []

    def _record_progress(**_kwargs):
        raise AssertionError("progress should not be recorded for duplicate records")

    global_generated, local_generated, produced_this_library, duplicate_records, accepted = persist_candidate_solution(
        solution_output_context=object(),
        progress_context=object(),
        sol=object(),
        seq="ACGT",
        final_seq="ACGT",
        used_tfbs=[],
        used_tfbs_detail=[],
        used_tf_counts={},
        used_tf_list=[],
        pad_meta={},
        covers_all=True,
        covers_required=True,
        tfbs_parts=[],
        regulator_labels=[],
        library_for_opt=[],
        sampling_info={},
        required_regulators=[],
        min_required_regulators=None,
        sampling_fraction=None,
        sampling_fraction_pairs=None,
        sampling_library_index=1,
        sampling_library_hash="h1",
        library_tfbs=[],
        library_tfs=[],
        library_site_ids=[],
        library_sources=[],
        promoter_detail={},
        sequence_validation={},
        solver_status="optimal",
        solver_objective=1.0,
        solver_solve_time_s=0.1,
        apply_pad_offsets=lambda detail, _pad: detail,
        global_generated=2,
        local_generated=1,
        produced_this_library=1,
        duplicate_records=4,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        write_output_fn=_write_output,
        record_progress_fn=_record_progress,
    )
    assert accepted is False
    assert (global_generated, local_generated, produced_this_library, duplicate_records) == (2, 1, 1, 5)


def test_persist_candidate_solution_records_progress_on_accept() -> None:
    def _write_output(**_kwargs):
        return "accepted", [{"tf": "lexA"}]

    def _record_progress(**_kwargs):
        return 9, 7, 6

    global_generated, local_generated, produced_this_library, duplicate_records, accepted = persist_candidate_solution(
        solution_output_context=object(),
        progress_context=object(),
        sol=object(),
        seq="ACGT",
        final_seq="ACGT",
        used_tfbs=[],
        used_tfbs_detail=[],
        used_tf_counts={},
        used_tf_list=[],
        pad_meta={},
        covers_all=True,
        covers_required=True,
        tfbs_parts=[],
        regulator_labels=[],
        library_for_opt=[],
        sampling_info={},
        required_regulators=[],
        min_required_regulators=None,
        sampling_fraction=None,
        sampling_fraction_pairs=None,
        sampling_library_index=1,
        sampling_library_hash="h1",
        library_tfbs=[],
        library_tfs=[],
        library_site_ids=[],
        library_sources=[],
        promoter_detail={},
        sequence_validation={},
        solver_status="optimal",
        solver_objective=1.0,
        solver_solve_time_s=0.1,
        apply_pad_offsets=lambda detail, _pad: detail,
        global_generated=2,
        local_generated=1,
        produced_this_library=1,
        duplicate_records=4,
        duplicate_solutions=0,
        failed_solutions=0,
        stall_events=0,
        write_output_fn=_write_output,
        record_progress_fn=_record_progress,
    )
    assert accepted is True
    assert (global_generated, local_generated, produced_this_library, duplicate_records) == (9, 7, 6, 4)
