"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/solution_outputs.py

Solution output recording helpers for Stage-B generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from ...adapters.outputs import OutputRecord, SinkBase
from .attempts import _append_attempt
from .outputs import _emit_event, _write_to_sinks

log = logging.getLogger(__name__)


def record_solution_outputs(
    *,
    sinks: list[SinkBase],
    final_seq: str,
    derived: dict,
    source_label: str,
    plan_name: str,
    output_bio_type: str,
    output_alphabet: str,
    tables_root: Path,
    run_id: str,
    next_attempt_index: Callable[[], int],
    used_tf_counts: dict[str, int],
    used_tf_list: list[str],
    sampling_library_index: int,
    sampling_library_hash: str,
    solver_status: str | None,
    solver_objective: float | None,
    solver_solve_time_s: float | None,
    dense_arrays_version: str | None,
    dense_arrays_version_source: str,
    library_tfbs: list[str],
    library_tfs: list[str],
    library_site_ids: list[str | None],
    library_sources: list[str],
    attempts_buffer: list[dict],
    solution_rows: list[dict] | None,
    composition_rows: list[dict] | None,
    events_path: Path | None,
    used_tfbs: list[str],
    used_tfbs_detail: list[dict],
) -> bool:
    record = OutputRecord.from_sequence(
        sequence=final_seq,
        meta=derived,
        source=source_label,
        bio_type=output_bio_type,
        alphabet=output_alphabet,
    )

    if not _write_to_sinks(sinks, record):
        return False

    attempts_buffer.append(
        {
            "created_at": derived.get("created_at"),
            "input_name": source_label,
            "plan_name": plan_name,
            "attempt_index": next_attempt_index(),
            "status": "ok",
            "reason": None,
            "detail": None,
            "sequence": final_seq,
            "used_tf_counts": used_tf_counts,
            "used_tf_list": used_tf_list,
            "sampling_library_index": int(sampling_library_index),
            "sampling_library_hash": str(sampling_library_hash),
            "solver_status": solver_status,
            "solver_objective": solver_objective,
            "solver_solve_time_s": solver_solve_time_s,
            "dense_arrays_version": dense_arrays_version,
            "dense_arrays_version_source": dense_arrays_version_source,
            "library_tfbs": library_tfbs,
            "library_tfs": library_tfs,
            "library_site_ids": library_site_ids,
            "library_sources": library_sources,
        }
    )
    if solution_rows is not None:
        solution_rows.append(
            _append_attempt(
                tables_root,
                run_id=run_id,
                input_name=source_label,
                plan_name=plan_name,
                attempt_index=next_attempt_index(),
                status="ok",
                reason=None,
                detail=None,
                sequence=final_seq,
                used_tf_counts=used_tf_counts,
                used_tf_list=used_tf_list,
                sampling_library_index=int(sampling_library_index),
                sampling_library_hash=str(sampling_library_hash),
                solver_status=solver_status,
                solver_objective=solver_objective,
                solver_solve_time_s=solver_solve_time_s,
                dense_arrays_version=dense_arrays_version,
                dense_arrays_version_source=dense_arrays_version_source,
                library_tfbs=library_tfbs,
                library_tfs=library_tfs,
                library_site_ids=library_site_ids,
                library_sources=library_sources,
                attempts_buffer=attempts_buffer,
            )
        )
    if composition_rows is not None:
        solution_id = record.id
        for entry in used_tfbs_detail:
            row = {
                "solution_id": solution_id,
                "input_name": source_label,
                "plan_name": plan_name,
                "library_index": int(sampling_library_index),
                "library_hash": str(sampling_library_hash),
            }
            if isinstance(entry, dict):
                row.update(entry)
            composition_rows.append(row)
    if events_path is not None:
        try:
            _emit_event(
                events_path,
                event="SOLUTION_ACCEPTED",
                payload={
                    "input_name": source_label,
                    "plan_name": plan_name,
                    "sequence": final_seq,
                    "library_index": int(sampling_library_index),
                    "library_hash": str(sampling_library_hash),
                },
            )
        except Exception:
            log.debug("Failed to emit SOLUTION_ACCEPTED event.", exc_info=True)

    return True
