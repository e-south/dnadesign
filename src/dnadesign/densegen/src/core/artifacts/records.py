"""
Typed record definitions for attempt/solution artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ids import hash_attempt_id


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    attempt_index: int
    run_id: str
    input_name: str
    plan_name: str
    created_at: str
    status: str
    reason: str
    detail_json: str
    sequence: str
    sequence_hash: str
    solution_id: str | None
    used_tf_counts_json: str
    used_tf_list: list[str]
    sampling_library_index: int
    sampling_library_hash: str
    solver_status: str | None
    solver_objective: float | None
    solver_solve_time_s: float | None
    dense_arrays_version: str | None
    dense_arrays_version_source: str
    library_tfbs: list[str]
    library_tfs: list[str]
    library_site_ids: list[str]
    library_sources: list[str]

    @classmethod
    def build(
        cls,
        *,
        attempt_index: int,
        run_id: str,
        input_name: str,
        plan_name: str,
        created_at: str,
        status: str,
        reason: str,
        detail_json: str,
        sequence: str,
        sequence_hash: str,
        solution_id: str | None,
        used_tf_counts_json: str,
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
        library_site_ids: list[str],
        library_sources: list[str],
    ) -> "AttemptRecord":
        attempt_id = hash_attempt_id(
            run_id=run_id,
            input_name=input_name,
            plan_name=plan_name,
            library_hash=sampling_library_hash,
            attempt_index=int(attempt_index),
        )
        return cls(
            attempt_id=attempt_id,
            attempt_index=int(attempt_index),
            run_id=str(run_id),
            input_name=str(input_name),
            plan_name=str(plan_name),
            created_at=str(created_at),
            status=str(status),
            reason=str(reason),
            detail_json=str(detail_json),
            sequence=str(sequence),
            sequence_hash=str(sequence_hash),
            solution_id=str(solution_id) if solution_id is not None else None,
            used_tf_counts_json=str(used_tf_counts_json),
            used_tf_list=list(used_tf_list or []),
            sampling_library_index=int(sampling_library_index),
            sampling_library_hash=str(sampling_library_hash),
            solver_status=str(solver_status) if solver_status is not None else None,
            solver_objective=float(solver_objective) if solver_objective is not None else None,
            solver_solve_time_s=float(solver_solve_time_s) if solver_solve_time_s is not None else None,
            dense_arrays_version=str(dense_arrays_version) if dense_arrays_version is not None else None,
            dense_arrays_version_source=str(dense_arrays_version_source),
            library_tfbs=list(library_tfbs or []),
            library_tfs=list(library_tfs or []),
            library_site_ids=list(library_site_ids or []),
            library_sources=list(library_sources or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "attempt_index": int(self.attempt_index),
            "run_id": self.run_id,
            "input_name": self.input_name,
            "plan_name": self.plan_name,
            "created_at": self.created_at,
            "status": self.status,
            "reason": self.reason,
            "detail_json": self.detail_json,
            "sequence": self.sequence,
            "sequence_hash": self.sequence_hash,
            "solution_id": self.solution_id,
            "used_tf_counts_json": self.used_tf_counts_json,
            "used_tf_list": list(self.used_tf_list),
            "sampling_library_index": int(self.sampling_library_index),
            "sampling_library_hash": self.sampling_library_hash,
            "solver_status": self.solver_status,
            "solver_objective": self.solver_objective,
            "solver_solve_time_s": self.solver_solve_time_s,
            "dense_arrays_version": self.dense_arrays_version,
            "dense_arrays_version_source": self.dense_arrays_version_source,
            "library_tfbs": list(self.library_tfbs),
            "library_tfs": list(self.library_tfs),
            "library_site_ids": list(self.library_site_ids),
            "library_sources": list(self.library_sources),
        }


@dataclass(frozen=True)
class SolutionRecord:
    solution_id: str
    attempt_id: str
    run_id: str
    input_name: str
    plan_name: str
    created_at: str
    sequence: str
    sequence_hash: str
    sampling_library_index: int
    sampling_library_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution_id": self.solution_id,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "input_name": self.input_name,
            "plan_name": self.plan_name,
            "created_at": self.created_at,
            "sequence": self.sequence,
            "sequence_hash": self.sequence_hash,
            "sampling_library_index": int(self.sampling_library_index),
            "sampling_library_hash": self.sampling_library_hash,
        }
