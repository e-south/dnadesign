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
    table = pq.read_table(table_path)
    available_columns = [column for column in selected_columns if column in table.column_names]
    rows = table.select(available_columns).to_pylist()
    title = html.escape(str(row.get("title") or "Six Eco1 variants selected for assay review"))
    return mo.vstack(
        [
            mo.Html(f"<h3 style='margin:0 0 0.35rem 0; font-size:1.08rem;'>{title}</h3>"),
            mo.ui.table([_display_row(raw) for raw in rows], page_size=10),
            _candidate_reason_accordion(rows, mo=mo),
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
        "MSA observed fraction": _round_or_none(trace.get("selection_support_alt_observed_fraction")),
        "unobserved MSA changes": _int_or_none(trace.get("selection_support_unobserved_mutation_count")),
        "NA-facing mutations": trace.get("nucleic_acid_facing_mutation_count"),
        "NA-facing charge change": trace.get("nucleic_acid_facing_charge_delta"),
        "chemistry warnings": trace.get("nucleic_acid_facing_chemistry_warning_count"),
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


def _candidate_reason_accordion(rows: list[dict[str, Any]], *, mo: Any) -> Any:
    panels: dict[str, Any] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        trace = _parse_trace(row.get("tie_break_trace_json"))
        panels[f"Why this row: {candidate_id}"] = mo.md(_candidate_reason_text(row=row, trace=trace))
    return mo.accordion(panels, multiple=False, lazy=True)


def _candidate_reason_text(*, row: dict[str, Any], trace: dict[str, object]) -> str:
    lines = [
        str(row.get("selection_reason") or "").strip(),
        f"MSA observed fraction: {_display_metric(trace.get('selection_support_alt_observed_fraction'))}",
        f"Unobserved MSA changes: {_display_metric(trace.get('selection_support_unobserved_mutation_count'))}",
        f"Nucleic-acid-facing mutations: {_display_metric(trace.get('nucleic_acid_facing_mutation_count'))}",
        f"Nucleic-acid-facing charge change: {_display_metric(trace.get('nucleic_acid_facing_charge_delta'))}",
        f"Chemistry warnings: {_display_metric(trace.get('nucleic_acid_facing_chemistry_warning_count'))}",
        f"Distal scaffold changes: {_display_metric(trace.get('distal_scaffold_mutation_count'))}",
        f"Nearest selected distance: {_display_metric(trace.get('nearest_selected_distance_aa'))}",
    ]
    return "\n".join(f"- {line}" for line in lines if line)


def _display_metric(value: object) -> str:
    if value is None:
        return "n/a"
    rounded = _round_or_none(value)
    if rounded is not None:
        if rounded.is_integer():
            return str(int(rounded))
        return str(rounded)
    return str(value)


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
