"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/stage_b_runtime_checks.py

Stage-B runtime checks for solver requirements, padding, and sequence constraints.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..sequence_constraints import CompiledSequenceConstraints, validate_sequence_constraints
from .stage_b_runtime_types import SequenceConstraintEvaluation


def _evaluate_solution_requirements(
    *,
    min_count_per_tf: int,
    tf_list_from_library: list[str],
    required_regulators: list[str],
    plan_min_count_by_regulator: dict[str, int],
    used_tf_counts: dict[str, int],
) -> tuple[bool, bool, str | None, dict]:
    covers_all = True
    covers_required = True
    if min_count_per_tf > 0 and tf_list_from_library:
        missing = [tf for tf in tf_list_from_library if used_tf_counts.get(tf, 0) < min_count_per_tf]
        if missing:
            return (
                False,
                covers_required,
                "min_count_per_tf",
                {
                    "min_count_per_tf": int(min_count_per_tf),
                    "missing_tfs": missing,
                },
            )
    if required_regulators:
        missing = [tf for tf in required_regulators if used_tf_counts.get(tf, 0) < 1]
        if missing:
            return (
                covers_all,
                False,
                "required_regulators",
                {
                    "required_regulators": list(required_regulators),
                    "missing_tfs": missing,
                },
            )
    if plan_min_count_by_regulator:
        missing = [
            tf for tf, min_count in plan_min_count_by_regulator.items() if used_tf_counts.get(tf, 0) < int(min_count)
        ]
        if missing:
            return (
                covers_all,
                covers_required,
                "min_count_by_regulator",
                {
                    "min_count_by_regulator": [
                        {
                            "tf": tf,
                            "min_count": int(plan_min_count_by_regulator[tf]),
                            "found": int(used_tf_counts.get(tf, 0)),
                        }
                        for tf in missing
                    ]
                },
            )
    return covers_all, covers_required, None, {}


def _extract_solver_metrics(sol) -> tuple[str | None, float | None, float | None]:
    solver_status = getattr(sol, "status", None)
    if solver_status is not None:
        solver_status = str(solver_status)
    solver_objective = getattr(sol, "objective", None)
    if solver_objective is None:
        solver_objective = getattr(sol, "objective_value", None)
    try:
        solver_objective = float(solver_objective) if solver_objective is not None else None
    except (TypeError, ValueError):
        solver_objective = None
    solver_solve_time_s = getattr(sol, "_densegen_solve_time_s", None)
    return solver_status, solver_objective, solver_solve_time_s


def _maybe_pad_sequence(
    *,
    sequence: str,
    seq_len: int,
    source_label: str,
    plan_name: str,
    pad_enabled: bool,
    pad_end: str,
    pad_mode: str,
    pad_gc_mode: str,
    pad_gc_min: float,
    pad_gc_max: float,
    pad_gc_target: float,
    pad_gc_tolerance: float,
    pad_gc_min_length: int,
    pad_max_tries: int,
    sequence_constraint_patterns: list[str],
    pad_builder,
    rng,
) -> tuple[str, dict]:
    final_seq = sequence
    pad_meta = {"used": False}
    if not pad_enabled and len(final_seq) < seq_len:
        raise RuntimeError(f"[{source_label}/{plan_name}] Sequence shorter than target and pad.mode=off.")
    if not pad_enabled or len(final_seq) >= seq_len:
        return final_seq, pad_meta

    gap = seq_len - len(final_seq)
    pad_forbidden_kmers = sorted(set(sequence_constraint_patterns))
    pad_left_context = "" if pad_end == "5prime" else final_seq
    pad_right_context = final_seq if pad_end == "5prime" else ""
    padded = pad_builder(
        length=gap,
        mode=pad_mode,
        gc_mode=pad_gc_mode,
        gc_min=pad_gc_min,
        gc_max=pad_gc_max,
        gc_target=pad_gc_target,
        gc_tolerance=pad_gc_tolerance,
        gc_min_pad_length=pad_gc_min_length,
        max_tries=pad_max_tries,
        forbid_kmers=pad_forbidden_kmers if pad_forbidden_kmers else None,
        left_context=pad_left_context,
        right_context=pad_right_context,
        rng=rng,
    )
    if isinstance(padded, tuple) and len(padded) == 2:
        pad_seq, pad_info = padded
        pad_info = pad_info or {}
    else:
        pad_seq, pad_info = padded, {}

    final_seq = (pad_seq + final_seq) if pad_end == "5prime" else (final_seq + pad_seq)
    pad_meta = {
        "used": True,
        "bases": gap,
        "end": pad_end,
        "gc_mode": pad_info.get("gc_mode", pad_gc_mode),
        "gc_min": pad_info.get("final_gc_min"),
        "gc_max": pad_info.get("final_gc_max"),
        "gc_target_min": pad_info.get("target_gc_min"),
        "gc_target_max": pad_info.get("target_gc_max"),
        "gc_actual": pad_info.get("gc_actual"),
        "relaxed": pad_info.get("relaxed"),
        "relaxed_reason": pad_info.get("relaxed_reason"),
        "attempts": pad_info.get("attempts"),
    }
    return final_seq, pad_meta


def _evaluate_sequence_constraints(
    *,
    final_seq: str,
    compiled_sequence_constraints,
    fixed_elements_dump: dict,
    source_label: str,
    plan_name: str,
    sampling_library_index: int,
    sampling_library_hash: str,
) -> SequenceConstraintEvaluation:
    promoter_detail = {"placements": []}
    sequence_validation = {"validation_passed": True, "violations": []}
    fixed_elements_dump = dict(fixed_elements_dump or {})
    has_promoter_constraints = bool(list(fixed_elements_dump.get("promoter_constraints") or []))
    compiled = compiled_sequence_constraints
    if compiled is None:
        if not has_promoter_constraints:
            return SequenceConstraintEvaluation(
                promoter_detail=promoter_detail,
                sequence_validation=sequence_validation,
                rejection_detail=None,
                rejection_event_payload=None,
                error=None,
            )
        compiled = CompiledSequenceConstraints(
            forbid_rules=(),
            allow_components=("upstream", "downstream"),
            generation_forbidden_patterns=(),
        )
    elif not compiled.has_rules() and not has_promoter_constraints:
        return SequenceConstraintEvaluation(
            promoter_detail=promoter_detail,
            sequence_validation=sequence_validation,
            rejection_detail=None,
            rejection_event_payload=None,
            error=None,
        )
    try:
        validation_result = validate_sequence_constraints(
            sequence=final_seq,
            compiled=compiled,
            fixed_elements_dump=fixed_elements_dump,
        )
    except ValueError as exc:
        error_message = str(exc)
        return SequenceConstraintEvaluation(
            promoter_detail=promoter_detail,
            sequence_validation={"validation_passed": False, "violations": []},
            rejection_detail={"error": error_message},
            rejection_event_payload={
                "input_name": source_label,
                "plan_name": plan_name,
                "library_index": int(sampling_library_index),
                "library_hash": str(sampling_library_hash),
                "error": error_message,
            },
            error=exc,
        )
    except Exception as exc:
        raise RuntimeError(f"[{source_label}/{plan_name}] sequence constraint evaluation failed: {exc}") from exc
    promoter_detail = dict(validation_result.promoter_detail or {"placements": []})
    sequence_validation = {
        "validation_passed": bool(validation_result.validation_passed),
        "violations": list(validation_result.violations or []),
    }
    if bool(validation_result.validation_passed):
        return SequenceConstraintEvaluation(
            promoter_detail=promoter_detail,
            sequence_validation=sequence_validation,
            rejection_detail=None,
            rejection_event_payload=None,
            error=None,
        )
    violations = list(validation_result.violations or [])
    return SequenceConstraintEvaluation(
        promoter_detail=promoter_detail,
        sequence_validation=sequence_validation,
        rejection_detail={"violations": violations},
        rejection_event_payload={
            "input_name": source_label,
            "plan_name": plan_name,
            "library_index": int(sampling_library_index),
            "library_hash": str(sampling_library_hash),
            "violations": violations,
        },
        error=None,
    )
