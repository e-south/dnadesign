"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/candidate_generation.py

Deterministic candidate enumeration for YIU v4 optimization.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

from dnadesign.cruncher.yiu.errors import YIU_JUNCTION_INVALID, NoFeasiblePlanError, raise_yiu_error
from dnadesign.cruncher.yiu.spec_models import JunctionOptimizationSpec, MismatchesSpec

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


@dataclass(frozen=True)
class CandidateWindowSummary:
    valid_internal_windows: int
    feasible_windows: int


def deterministic_fallback_mutated_base(*, native_base: str, opposing_base: str) -> str:
    if opposing_base in _BASES and opposing_base != native_base:
        return opposing_base
    for base in _BASES:
        if base != native_base:
            return base
    raise ValueError("unable to choose a deterministic fallback mutated base")


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
) -> tuple[tuple[int, ...], CandidateWindowSummary]:
    valid_internal = internal_window_starts(payload_length, overhang_length=junction_spec.overhang_length)
    if not valid_internal:
        raise_yiu_error(
            YIU_JUNCTION_INVALID,
            f"payload length {payload_length} is too short for any internal 4-nt junction window",
        )
    if junction_spec.mode == "explicit_window":
        assert junction_spec.start is not None
        if junction_spec.start not in valid_internal:
            raise_yiu_error(
                YIU_JUNCTION_INVALID,
                "optimization.junction.start/end must define an internal 4-nt window with non-empty left/right bodies",
            )
        return (junction_spec.start,), CandidateWindowSummary(len(valid_internal), 1)
    if junction_spec.mode == "derived":
        midpoint = payload_length
        winner = min(valid_internal, key=lambda start: (abs((start + (start + 4)) - midpoint), start))
        return (winner,), CandidateWindowSummary(len(valid_internal), 1)

    feasible = tuple(
        start
        for start in valid_internal
        if start <= junction_spec.max_payload_body_length
        and (payload_length - (start + 4)) <= junction_spec.max_payload_body_length
    )
    if not feasible:
        raise NoFeasiblePlanError(
            "No feasible optimized junction found for "
            f"payload length {payload_length} under max_payload_body_length={junction_spec.max_payload_body_length}. "
            f"Required internal 4-nt windows exist: {len(valid_internal)}. "
            "Windows satisfying the body-length constraint: 0. "
            "Reduce payload length, relax the body-length bound, or use an explicit internal window."
        )
    return feasible, CandidateWindowSummary(len(valid_internal), len(feasible))


def _base_candidates(
    *,
    native_base: str,
    opposing_base: str,
    pwm_effective: bool,
) -> tuple[str, ...]:
    if pwm_effective:
        return tuple(base for base in _BASES if base != native_base)
    return (deterministic_fallback_mutated_base(native_base=native_base, opposing_base=opposing_base),)


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
                    base_options.append(
                        _base_candidates(
                            native_base=native_base,
                            opposing_base=opposing_base,
                            pwm_effective=pwm_effective,
                        )
                    )
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
