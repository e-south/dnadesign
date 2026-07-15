"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/selection_summary.py

Selection-summary helpers for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    canonical_mutation_positions,
    canonical_mutation_tokens,
    jaccard_distance,
)


def build_selection_summary(
    *,
    triage_rows: Sequence[dict[str, object]],
    local_structure_rows: Sequence[dict[str, object]],
    panel_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Return manifest-ready summary fields derived from materialized selector rows."""

    return {
        "candidate_counts": {
            "candidate_pool": len(triage_rows),
            "selection_contract_pass": sum(bool(row.get("selection_contract_pass")) for row in triage_rows),
            "selection_contract_pass_by_policy": _count_by(
                [row for row in triage_rows if bool(row.get("selection_contract_pass"))],
                "policy_id",
            ),
            "wang_r13a_interface_disruption_evidence_match": sum(
                bool(row.get("wang_r13a_interface_disruption_evidence_match")) for row in triage_rows
            ),
            "selected_panel": len(panel_rows),
        },
        "gate_counts": {
            "hard_gate_status": _count_by(triage_rows, "hard_gate_status"),
            "local_structure_gate_status": _count_by(triage_rows, "local_structure_gate_status"),
            "wang_alpha1_r13_review_status": _count_by(triage_rows, "wang_alpha1_r13_review_status"),
            "rt_msdna_oligomeric_state_review_status": _count_by(
                triage_rows, "rt_msdna_oligomeric_state_review_status"
            ),
        },
        "local_structure_region_threshold_counts": _nested_count_by(
            local_structure_rows,
            outer_key="region_id",
            inner_key="local_ca_rmsd_threshold_status",
        ),
        "selected_generation_policy_counts": _count_by(panel_rows, "policy_id"),
        "selected_mutation_overlap": _selected_mutation_overlap(
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
        ),
        "selected_mutation_overlap_by_policy": {
            policy_id: _selected_mutation_overlap(
                panel_rows=[row for row in panel_rows if str(row.get("policy_id") or "") == policy_id],
                candidate_rows=candidate_rows,
            )
            for policy_id in sorted({str(row.get("policy_id") or "") for row in panel_rows})
        },
    }


def _selected_mutation_overlap(
    *,
    panel_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    candidate_by_id = {str(row.get("candidate_id") or ""): row for row in candidate_rows}
    token_sets = [
        canonical_mutation_tokens(
            candidate_by_id.get(str(row.get("candidate_id") or ""), {}).get("canonical_mutations")
        )
        for row in panel_rows
    ]
    position_sets = [
        canonical_mutation_positions(
            candidate_by_id.get(str(row.get("candidate_id") or ""), {}).get("canonical_mutations")
        )
        for row in panel_rows
    ]
    shared_tokens = sorted(set.intersection(*map(set, token_sets))) if token_sets else []
    shared_positions = sorted(set.intersection(*map(set, position_sets))) if position_sets else []
    position_distances = [jaccard_distance(left, right) for left, right in combinations(position_sets, 2)]
    token_distances = [jaccard_distance(left, right) for left, right in combinations(token_sets, 2)]
    return {
        "selected_candidate_count": len(panel_rows),
        "unique_exact_substitution_count": len(set().union(*token_sets)) if token_sets else 0,
        "shared_exact_substitution_count": len(shared_tokens),
        "shared_exact_substitutions": shared_tokens,
        "unique_mutated_position_count": len(set().union(*position_sets)) if position_sets else 0,
        "shared_mutated_position_count": len(shared_positions),
        "shared_mutated_positions": shared_positions,
        "mean_pairwise_mutated_position_jaccard_distance": _mean_or_none(position_distances),
        "minimum_pairwise_mutated_position_jaccard_distance": _minimum_or_none(position_distances),
        "mean_pairwise_exact_substitution_jaccard_distance": _mean_or_none(token_distances),
        "minimum_pairwise_exact_substitution_jaccard_distance": _minimum_or_none(token_distances),
        "mutation_count_range": _integer_range(panel_rows, "mutation_count_total"),
        "peripheral_mutation_count_range": _integer_range(panel_rows, "nucleic_acid_facing_mutation_count"),
        "peripheral_charge_change_range": _integer_range(panel_rows, "nucleic_acid_facing_charge_delta"),
    }


def _integer_range(rows: Sequence[dict[str, object]], key: str) -> list[int]:
    values = [int(row[key]) for row in rows if row.get(key) is not None]
    return [min(values), max(values)] if values else []


def _mean_or_none(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _minimum_or_none(values: Sequence[float]) -> float | None:
    return round(min(values), 3) if values else None


def _count_by(rows: Sequence[dict[str, object]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "missing") for row in rows)
    return {value: counts[value] for value in sorted(counts)}


def _nested_count_by(
    rows: Sequence[dict[str, object]],
    *,
    outer_key: str,
    inner_key: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        outer_value = str(row.get(outer_key) or "missing")
        inner_value = str(row.get(inner_key) or "missing")
        counts.setdefault(outer_value, Counter())[inner_value] += 1
    return {
        outer_value: {inner_value: counter[inner_value] for inner_value in sorted(counter)}
        for outer_value, counter in sorted(counts.items())
    }


__all__ = ["build_selection_summary"]
