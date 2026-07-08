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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    canonical_mutation_positions,
    canonical_mutation_tokens,
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
            "preservation_pass": _count_value(triage_rows, "hard_gate_status", "eligible"),
            "chemistry_support_pass": _count_value(
                triage_rows,
                "selection_candidate_tier",
                "primary_panel_candidate",
            ),
            "selected_panel": len(panel_rows),
        },
        "gate_counts": {
            "hard_gate_status": _count_by(triage_rows, "hard_gate_status"),
            "local_structure_gate_status": _count_by(triage_rows, "local_structure_gate_status"),
            "selection_candidate_tier": _count_by(triage_rows, "selection_candidate_tier"),
        },
        "local_structure_region_threshold_counts": _nested_count_by(
            local_structure_rows,
            outer_key="region_id",
            inner_key="local_ca_rmsd_threshold_status",
        ),
        "selected_design_class_counts": _count_by(panel_rows, "design_class_id"),
        "selected_mutation_overlap": _selected_mutation_overlap(
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
        ),
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
    return {
        "selected_candidate_count": len(panel_rows),
        "shared_exact_substitution_count": len(shared_tokens),
        "shared_exact_substitutions": shared_tokens,
        "shared_mutated_position_count": len(shared_positions),
        "shared_mutated_positions": shared_positions,
    }


def _count_by(rows: Sequence[dict[str, object]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "missing") for row in rows)
    return {value: counts[value] for value in sorted(counts)}


def _count_value(rows: Sequence[dict[str, object]], key: str, value: str) -> int:
    return sum(1 for row in rows if str(row.get(key) or "") == value)


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
