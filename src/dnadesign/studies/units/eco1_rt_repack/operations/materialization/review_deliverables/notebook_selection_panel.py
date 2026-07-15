"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_selection_panel.py

Selection-panel rendering helpers for the Eco1 review-deliverables notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def render_selection_panel_table(row: dict[str, Any], *, mo: Any, table_path: Path) -> Any:
    """Render the selected Eco1 panel table as a compact marimo table."""

    if not table_path.exists():
        return mo.md(f"Selection panel table unavailable: `{table_path}`")
    selected_columns = [
        "selection_slot",
        "selection_rank",
        "design_group_id",
        "within_group_rank",
        "candidate_id",
        "policy_id",
        "fold_review_class",
        "mutation_count_total",
        "mean_plddt",
        "within_group_nearest_mutated_position_jaccard_distance",
        "within_group_nearest_exact_substitution_jaccard_distance",
        "selection_support_alt_observed_fraction",
        "nucleic_acid_facing_mutation_count",
        "nucleic_acid_facing_charge_delta",
        "nucleic_acid_facing_basic_gain_count",
        "nucleic_acid_facing_basic_loss_count",
        "nucleic_acid_facing_acidic_gain_count",
        "thumb_contact_track_mutation_count",
        "c_terminal_primer_rna_recognition_mutation_count",
        "wang_alpha1_f10_substitution",
        "wang_alpha1_r13_substitution",
        "wang_r13a_interface_disruption_evidence_match",
        "rt_msdna_oligomeric_state_review_status",
        "wang_alpha1_mutation_count",
    ]
    parquet_file = pq.ParquetFile(table_path)
    missing_columns = [column for column in selected_columns if column not in parquet_file.schema.names]
    if missing_columns:
        raise ValueError(f"Selection panel table is missing notebook columns: {', '.join(missing_columns)}")
    rows = parquet_file.read(columns=selected_columns).to_pylist()
    title = html.escape(str(row.get("title") or "Selected Eco1 RT panel"))
    return mo.vstack(
        [
            mo.Html(f"<h3 style='margin:0 0 0.35rem 0; font-size:1.08rem;'>{title}</h3>"),
            mo.ui.table([_display_row(raw) for raw in rows], page_size=10),
        ],
        gap=0.25,
    )


def _display_row(row: dict[str, Any]) -> dict[str, object]:
    return {
        "slot": str(row.get("selection_slot") or ""),
        "selection rank": _int_or_none(row.get("selection_rank")),
        "design group": str(row.get("design_group_id") or ""),
        "within-group rank": _int_or_none(row.get("within_group_rank")),
        "candidate": str(row.get("candidate_id") or "").removeprefix("thread_candidate_"),
        "policy": str(row.get("policy_id") or ""),
        "fold": str(row.get("fold_review_class") or ""),
        "mutations": _int_or_none(row.get("mutation_count_total")),
        "pLDDT": _round_or_none(row.get("mean_plddt")),
        "within-policy mutation-position distance": _round_or_none(
            row.get("within_group_nearest_mutated_position_jaccard_distance")
        ),
        "within-policy exact-substitution distance": _round_or_none(
            row.get("within_group_nearest_exact_substitution_jaccard_distance")
        ),
        "MSA observed fraction": _round_or_none(row.get("selection_support_alt_observed_fraction")),
        "peripheral DNA/RNA edits": _int_or_none(row.get("nucleic_acid_facing_mutation_count")),
        "peripheral charge change": _int_or_none(row.get("nucleic_acid_facing_charge_delta")),
        "basic gains": _int_or_none(row.get("nucleic_acid_facing_basic_gain_count")),
        "basic losses": _int_or_none(row.get("nucleic_acid_facing_basic_loss_count")),
        "acidic gains": _int_or_none(row.get("nucleic_acid_facing_acidic_gain_count")),
        "Wang thumb-track edits": _int_or_none(row.get("thumb_contact_track_mutation_count")),
        "residues 255-311 edits": _int_or_none(row.get("c_terminal_primer_rna_recognition_mutation_count")),
        "F10": str(row.get("wang_alpha1_f10_substitution") or ""),
        "R13": str(row.get("wang_alpha1_r13_substitution") or ""),
        "matches tested R13A": bool(row.get("wang_r13a_interface_disruption_evidence_match")),
        "RT-msDNA assembly state": str(row.get("rt_msdna_oligomeric_state_review_status") or ""),
        "alpha-1 mutations": _int_or_none(row.get("wang_alpha1_mutation_count")),
    }


def _round_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
