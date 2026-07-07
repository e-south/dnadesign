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
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def render_selection_panel_table(row: dict[str, Any], *, mo: Any, table_path: Path) -> Any:
    """Render the selected Eco1 panel table as a compact marimo table."""

    if not table_path.exists():
        return mo.md(f"Selection panel table unavailable: `{table_path}`")
    selected_columns = [
        "selection_slot",
        "candidate_id",
        "fold_review_class",
        "feasibility_status",
        "mutation_count",
        "nearest_selected_distance_aa",
        "selection_reason",
        "tie_break_trace_json",
    ]
    parquet_file = pq.ParquetFile(table_path)
    available_columns = [column for column in selected_columns if column in parquet_file.schema.names]
    rows = parquet_file.read(columns=available_columns).to_pylist()
    title = html.escape(str(row.get("title") or "Six Eco1 RT variants form a protein review panel"))
    return mo.vstack(
        [
            mo.Html(f"<h3 style='margin:0 0 0.35rem 0; font-size:1.08rem;'>{title}</h3>"),
            mo.ui.table([_display_row(raw) for raw in rows], page_size=10),
        ],
        gap=0.25,
    )


def _display_row(row: dict[str, Any]) -> dict[str, object]:
    trace = _parse_trace(row.get("tie_break_trace_json"))
    return {
        "slot": str(row.get("selection_slot") or ""),
        "candidate": str(row.get("candidate_id") or "").removeprefix("thread_candidate_"),
        "fold": str(row.get("fold_review_class") or ""),
        "feasibility": str(row.get("feasibility_status") or ""),
        "mutations": _int_or_none(trace.get("mutation_count_total") or row.get("mutation_count")),
        "pLDDT": _round_or_none(trace.get("mean_plddt")),
        "WT RMSD A": _round_or_none(trace.get("wt_runtime_ca_rmsd")),
        "cryoEM RMSD A": _round_or_none(trace.get("cryoem_mapped_ca_rmsd")),
        "nearest selected distance": row.get("nearest_selected_distance_aa"),
        "nearest mutation-position distance": _round_or_none(
            trace.get("nearest_selected_mutation_position_jaccard_distance")
            or row.get("nearest_selected_mutation_position_jaccard_distance")
        ),
        "nearest exact-substitution distance": _round_or_none(
            trace.get("nearest_selected_mutation_token_jaccard_distance")
            or row.get("nearest_selected_mutation_token_jaccard_distance")
        ),
        "MSA observed fraction": _round_or_none(trace.get("selection_support_alt_observed_fraction")),
        "unobserved MSA changes": _int_or_none(trace.get("selection_support_unobserved_mutation_count")),
        "near retained DNA/RNA edits": _int_or_none(trace.get("nucleic_acid_facing_mutation_count")),
        "near-region charge change": _int_or_none(trace.get("nucleic_acid_facing_charge_delta")),
        "near-region chemistry warnings": _int_or_none(trace.get("nucleic_acid_facing_chemistry_warning_count")),
        "Wang thumb-track edits": _int_or_none(trace.get("thumb_contact_track_mutation_count")),
        "C-terminal primer-RNA edits": _int_or_none(trace.get("c_terminal_primer_rna_recognition_mutation_count")),
        "reason": str(row.get("selection_reason") or ""),
    }


def _parse_trace(value: object) -> dict[str, object]:
    if not value:
        return {}
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


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
