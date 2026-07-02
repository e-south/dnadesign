"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/feasibility.py

Feasibility report row construction for Eco1 RT candidate selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from ast import literal_eval
from collections.abc import Sequence

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    AMINO_ACIDS,
    CODON_POLICY_ID,
    CREATED_BY,
    FEASIBILITY_POLICY_ID,
)

_MUTATION_POSITION_RE = re.compile(r"[A-Z](\d+)[A-Z*]")


def build_feasibility_rows(
    *,
    candidate_rows: Sequence[dict[str, object]],
    foldcheck_report_rows: Sequence[dict[str, object]],
    input_candidate_pool_hash: str,
    input_mask_policy_hash: str,
    input_foldcheck_report_hash: str,
    created_at: str,
) -> list[dict[str, object]]:
    """Build computational feasibility rows for accepted synthetic candidates."""

    parent_hash = _parent_sequence_hash(foldcheck_report_rows)
    accepted_foldcheck_ids = {
        str(row["candidate_id"]) for row in foldcheck_report_rows if str(row.get("status")) == "accepted"
    }
    rows: list[dict[str, object]] = []
    for candidate in candidate_rows:
        if str(candidate.get("status")) != "accepted":
            continue
        candidate_id = str(candidate["candidate_id"])
        mutations = _as_strings(candidate.get("canonical_mutations"))
        mutation_positions = _mutation_positions(mutations)
        mutation_windows = _mutation_windows(mutation_positions)
        sequence = str(candidate.get("sequence") or "")
        protected_count = int(candidate.get("protected_mutation_count") or 0)
        outside_mutable = _as_ints(candidate.get("outside_mutable_positions"))
        blockers = _synthesis_blockers(
            sequence=sequence,
            protected_count=protected_count,
            outside_mutable=outside_mutable,
            has_foldcheck=candidate_id in accepted_foldcheck_ids,
        )
        mutation_count = int(candidate.get("mutation_count") or len(mutation_positions))
        tier = _synthesis_tier(mutation_count=mutation_count, blockers=blockers)
        rows.append(
            {
                "candidate_id": candidate_id,
                "sequence_hash": str(candidate["sequence_hash"]),
                "parent_sequence_id": "wild_type",
                "parent_sequence_hash": parent_hash,
                "mutation_count_total": mutation_count,
                "mutation_count_mutable_region": int(candidate.get("mutable_mutation_count") or mutation_count),
                "mutation_count_protected_region": protected_count,
                "protected_mutation_violation_count": protected_count,
                "protected_mutation_violations_json": json.dumps(outside_mutable, sort_keys=True),
                "mutation_windows_json": json.dumps(mutation_windows, sort_keys=True),
                "max_mutation_window_length": max((int(window["length"]) for window in mutation_windows), default=0),
                "max_mutation_window_mutation_count": max(
                    (int(window["mutation_count"]) for window in mutation_windows),
                    default=0,
                ),
                "mutation_window_density_max": max(
                    (float(window["density"]) for window in mutation_windows),
                    default=0.0,
                ),
                "nearest_parent_id": "wild_type",
                "nearest_parent_distance_aa": mutation_count,
                "nearest_parent_distance_fraction": _distance_fraction(mutation_count, sequence),
                "parent_haplotype_id": "wild_type_full_sequence",
                "parent_haplotype_distance_aa": mutation_count,
                "synthesis_tier": tier,
                "synthesis_blockers_json": json.dumps(blockers, sort_keys=True),
                "codon_policy_id": CODON_POLICY_ID,
                "sequence_complexity_flags_json": json.dumps(_sequence_flags(sequence), sort_keys=True),
                "feasibility_status": "blocked" if blockers else "feasible",
                "feasibility_reason": _feasibility_reason(blockers=blockers, tier=tier),
                "feasibility_policy_id": FEASIBILITY_POLICY_ID,
                "input_candidate_table_hash": input_candidate_pool_hash,
                "input_mask_policy_hash": input_mask_policy_hash,
                "input_foldcheck_report_hash": input_foldcheck_report_hash,
                "created_at_utc": created_at,
                "created_by": CREATED_BY,
            }
        )
    return rows


def _parent_sequence_hash(rows: Sequence[dict[str, object]]) -> str:
    for row in rows:
        if str(row.get("candidate_id")) == "wild_type":
            return str(row.get("input_sequence_hash") or row.get("sequence_hash") or "")
    return ""


def _synthesis_blockers(
    *,
    sequence: str,
    protected_count: int,
    outside_mutable: list[int],
    has_foldcheck: bool,
) -> list[str]:
    blockers: list[str] = []
    if protected_count:
        blockers.append("protected_mutation_violation")
    if outside_mutable:
        blockers.append("outside_mutable_position")
    if set(sequence) - AMINO_ACIDS:
        blockers.append("noncanonical_amino_acid")
    if not has_foldcheck:
        blockers.append("missing_accepted_foldcheck_row")
    return sorted(set(blockers))


def _synthesis_tier(*, mutation_count: int, blockers: list[str]) -> str:
    if blockers:
        return "blocked"
    if mutation_count <= 32:
        return "easy"
    if mutation_count <= 70:
        return "standard"
    return "difficult"


def _feasibility_reason(*, blockers: list[str], tier: str) -> str:
    if blockers:
        return "Blocked by " + ", ".join(blockers)
    return f"Computational full-gene candidate; synthesis tier is {tier}"


def _distance_fraction(mutation_count: int, sequence: str) -> float | None:
    return None if not sequence else mutation_count / len(sequence)


def _sequence_flags(sequence: str) -> list[str]:
    flags: list[str] = []
    if set(sequence) - AMINO_ACIDS:
        flags.append("noncanonical_amino_acid")
    if re.search(r"([KRH]){8,}", sequence):
        flags.append("polybasic_run_ge8")
    if re.search(r"([A-Z])\1{7,}", sequence):
        flags.append("homopolymer_run_ge8")
    return flags


def _mutation_positions(mutations: list[str]) -> list[int]:
    positions: list[int] = []
    for mutation in mutations:
        match = _MUTATION_POSITION_RE.search(mutation)
        if match:
            positions.append(int(match.group(1)))
    return sorted(set(positions))


def _mutation_windows(positions: list[int]) -> list[dict[str, object]]:
    if not positions:
        return []
    windows: list[dict[str, object]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        windows.append(_window(start, previous))
        start = previous = position
    windows.append(_window(start, previous))
    return windows


def _window(start: int, end: int) -> dict[str, object]:
    length = end - start + 1
    return {
        "start": start,
        "end": end,
        "length": length,
        "mutation_count": length,
        "density": 1.0,
    }


def _as_strings(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    text = str(value)
    try:
        parsed = literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list | tuple):
        return [str(item) for item in parsed]
    return [match.group(0) for match in _MUTATION_POSITION_RE.finditer(text)]


def _as_ints(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [int(item) for item in value]
    return [int(match) for match in re.findall(r"\d+", str(value))]
