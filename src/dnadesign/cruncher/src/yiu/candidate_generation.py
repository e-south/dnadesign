"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/candidate_generation.py

Deterministic candidate enumeration for YIU v4 optimization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

from dnadesign.cruncher.yiu.errors import YIU_JUNCTION_INVALID, NoFeasiblePlanError, raise_yiu_error
from dnadesign.cruncher.yiu.spec_rendering_models import JunctionOptimizationSpec, MismatchesSpec

_BASES = ("A", "C", "G", "T")


@dataclass(frozen=True)
class MutationChoice:
    junction_offset: int
    payload_index: int
    mutated_strand: str
    native_base: str
    mutated_base: str
    opposing_base: str


@dataclass(frozen=True)
class CandidatePlan:
    junction_start: int
    junction_end: int
    mismatch_positions: tuple[int, ...]
    mutations: tuple[MutationChoice, ...]
    midpoint_distance: int
    body_length_balance: int
    terminal_positions_used: int
    default_strand_preference_count: int
    lexical_key: str


def _body_lengths(*, payload_length: int, start: int, overhang_length: int) -> tuple[int, int]:
    left_body_length = start
    right_body_length = payload_length - (start + overhang_length)
    return left_body_length, right_body_length


def _window_satisfies_body_length_bound(
    *,
    payload_length: int,
    start: int,
    junction_spec: JunctionOptimizationSpec,
) -> bool:
    left_body_length, right_body_length = _body_lengths(
        payload_length=payload_length,
        start=start,
        overhang_length=junction_spec.overhang_length,
    )
    return (
        left_body_length <= junction_spec.max_payload_body_length
        and right_body_length <= junction_spec.max_payload_body_length
    )


def _raise_no_feasible_window_error(
    *,
    payload_length: int,
    junction_spec: JunctionOptimizationSpec,
    valid_internal_count: int,
    optimize_mode: bool,
) -> None:
    prefix = "No feasible optimized junction found for " if optimize_mode else "No feasible junction window found for "
    raise NoFeasiblePlanError(
        prefix
        + f"payload length {payload_length} under max_payload_body_length={junction_spec.max_payload_body_length}. "
        + f"Required internal 4-nt windows exist: {valid_internal_count}. "
        + "Windows satisfying the body-length constraint: 0. "
        + "Reduce payload length, relax the body-length bound, or use an explicit internal window."
    )


def internal_window_starts(payload_length: int, *, overhang_length: int = 4) -> tuple[int, ...]:
    if payload_length <= overhang_length + 1:
        return ()
    return tuple(
        start for start in range(1, payload_length - overhang_length) if (start + overhang_length) < payload_length
    )


def resolve_window_starts(
    *,
    payload_length: int,
    junction_spec: JunctionOptimizationSpec,
) -> tuple[int, ...]:
    valid_internal = internal_window_starts(payload_length, overhang_length=junction_spec.overhang_length)
    if not valid_internal:
        raise_yiu_error(
            YIU_JUNCTION_INVALID,
            f"payload length {payload_length} is too short for any internal 4-nt junction window",
        )
    feasible = tuple(
        start
        for start in valid_internal
        if _window_satisfies_body_length_bound(
            payload_length=payload_length,
            start=start,
            junction_spec=junction_spec,
        )
    )
    if junction_spec.canonical_mode == "explicit_window":
        assert junction_spec.start is not None
        if junction_spec.start not in valid_internal:
            raise_yiu_error(
                YIU_JUNCTION_INVALID,
                "optimization.junction.start/end must define an internal 4-nt window with non-empty left/right bodies",
            )
        if junction_spec.start not in feasible:
            left_body_length, right_body_length = _body_lengths(
                payload_length=payload_length,
                start=junction_spec.start,
                overhang_length=junction_spec.overhang_length,
            )
            raise_yiu_error(
                YIU_JUNCTION_INVALID,
                "optimization.junction.start/end must satisfy max_payload_body_length="
                f"{junction_spec.max_payload_body_length}; "
                f"got left_body_length={left_body_length}, right_body_length={right_body_length}",
            )
        return (junction_spec.start,)
    if junction_spec.canonical_mode == "center_locked":
        if not feasible:
            _raise_no_feasible_window_error(
                payload_length=payload_length,
                junction_spec=junction_spec,
                valid_internal_count=len(valid_internal),
                optimize_mode=False,
            )
        midpoint = payload_length
        winner = min(feasible, key=lambda start: (abs((start + (start + 4)) - midpoint), start))
        return (winner,)
    if not feasible:
        _raise_no_feasible_window_error(
            payload_length=payload_length,
            junction_spec=junction_spec,
            valid_internal_count=len(valid_internal),
            optimize_mode=True,
        )
    return feasible


def _base_candidates(*, native_base: str) -> tuple[str, ...]:
    return tuple(base for base in _BASES if base != native_base)


def enumerate_candidates(
    *,
    reference_payload_sequence: str,
    reference_complement_sequence: str,
    junction_starts: Iterable[int],
    mismatches_spec: MismatchesSpec,
    pwm_effective: bool,
) -> tuple[CandidatePlan, ...]:
    payload_length = len(reference_payload_sequence)
    ordered_strands = tuple(item for item in ("complement", "payload") if item in set(mismatches_spec.allowed_strands))
    plans: list[CandidatePlan] = []
    for junction_start in sorted(set(int(item) for item in junction_starts)):
        mismatch_sets = combinations(mismatches_spec.candidate_positions, mismatches_spec.count)
        for positions in mismatch_sets:
            for strands in product(ordered_strands, repeat=mismatches_spec.count):
                base_options: list[tuple[str, ...]] = []
                mutation_metadata: list[tuple[int, int, str, str, str]] = []
                for offset, strand in zip(positions, strands, strict=True):
                    payload_index = junction_start + offset
                    native_base = (
                        reference_payload_sequence[payload_index]
                        if strand == "payload"
                        else reference_complement_sequence[payload_index]
                    )
                    opposing_base = (
                        reference_complement_sequence[payload_index]
                        if strand == "payload"
                        else reference_payload_sequence[payload_index]
                    )
                    base_options.append(_base_candidates(native_base=native_base))
                    mutation_metadata.append((offset, payload_index, strand, native_base, opposing_base))
                for selected_bases in product(*base_options):
                    mutations = tuple(
                        MutationChoice(
                            junction_offset=offset,
                            payload_index=payload_index,
                            mutated_strand=strand,
                            native_base=native_base,
                            mutated_base=mutated_base,
                            opposing_base=opposing_base,
                        )
                        for (offset, payload_index, strand, native_base, opposing_base), mutated_base in zip(
                            mutation_metadata,
                            selected_bases,
                            strict=True,
                        )
                    )
                    lexical_key = "|".join(
                        [
                            str(junction_start),
                            ",".join(str(item) for item in positions),
                            ",".join(str(item) for item in strands),
                            ",".join(str(item) for item in selected_bases),
                        ]
                    )
                    plans.append(
                        CandidatePlan(
                            junction_start=junction_start,
                            junction_end=junction_start + 4,
                            mismatch_positions=tuple(positions),
                            mutations=mutations,
                            midpoint_distance=abs((junction_start + (junction_start + 4)) - payload_length),
                            body_length_balance=abs(junction_start - (payload_length - (junction_start + 4))),
                            terminal_positions_used=sum(1 for item in positions if item in {0, 3}),
                            default_strand_preference_count=sum(
                                1 for item in strands if item == mismatches_spec.default_strand_preference
                            ),
                            lexical_key=lexical_key,
                        )
                    )
    return tuple(plans)
