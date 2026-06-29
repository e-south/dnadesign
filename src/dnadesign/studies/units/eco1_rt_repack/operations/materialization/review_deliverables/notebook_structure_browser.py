"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_structure_browser.py

Interactive structure-browser helpers for the Eco1 review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    resolve_manifest_path,
)
from dnadesign.thread.structure_views import StructureViewModel, StructureViewSpec, render_structure_view_html


def load_structure_browser_rows(*, manifest_root: Path, deliverables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load interactive structure-browser rows if the manifest is materialized."""

    manifest_row = next(_interactive_structure_manifest_rows(deliverables), None)
    if manifest_row is None or str(manifest_row.get("status") or "") != "rendered":
        return []
    manifest_path = resolve_manifest_path(manifest_root, str(manifest_row["path"]))
    if not manifest_path.exists():
        return []
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    browser_root = manifest_path.parent
    reference = dict(payload.get("reference") or {})
    rows: list[dict[str, Any]] = []
    for row in payload.get("structures") or []:
        if not isinstance(row, dict):
            continue
        enriched = dict(row)
        enriched["_browser_root"] = str(browser_root)
        enriched["_reference"] = reference
        rows.append(enriched)
    return rows


def structure_browser_lookup(rows: list[dict[str, Any]], *, selected_section: str) -> dict[str, dict[str, Any]]:
    """Build structure dropdown labels for the fold-review section."""

    if selected_section != "fold_review":
        return {}
    return {_structure_browser_label(row): row for row in rows}


def render_structure_browser(
    *,
    mo: Any,
    selected_row: dict[str, Any] | None,
    structure_ui: Any,
) -> Any:
    """Render an interactive browser structure view for one selected fold model."""

    if selected_row is None:
        return mo.md("Interactive structure browsing is unavailable for this section.")
    browser_root = Path(str(selected_row["_browser_root"]))
    reference = dict(selected_row.get("_reference") or {})
    reference_path = resolve_manifest_path(browser_root, str(reference.get("local_path") or ""))
    query_path = resolve_manifest_path(browser_root, str(selected_row.get("local_path") or ""))
    if not reference_path.exists() or not query_path.exists():
        return mo.md("Interactive structure browsing is skipped because a selected PDB path is missing.")
    try:
        html_panel = render_structure_view_html(
            StructureViewSpec(
                title=_structure_browser_title(selected_row),
                models=(
                    StructureViewModel(
                        model_id=str(reference.get("model_id") or "reference"),
                        structure_text=reference_path.read_text(encoding="utf-8"),
                        label=str(reference.get("display_label") or "Reference"),
                        color=str(reference.get("color") or "#d8d8d8"),
                        opacity=0.82,
                    ),
                    StructureViewModel(
                        model_id=str(selected_row["candidate_id"]),
                        structure_text=query_path.read_text(encoding="utf-8"),
                        label=str(selected_row.get("display_label") or selected_row["candidate_id"]),
                        color=str(selected_row.get("color") or "#0072B2"),
                    ),
                ),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive notebook rendering path
        return mo.md(f"Interactive structure viewer failed to render: `{type(exc).__name__}: {exc}`")
    metric_rows = _structure_metric_rows(selected_row)
    return mo.vstack(
        [
            mo.hstack([structure_ui], justify="start", gap=1.0),
            mo.Html(html_panel),
            mo.ui.table(metric_rows, page_size=8),
        ],
        gap=0.35,
    )


def _interactive_structure_manifest_rows(deliverables: list[dict[str, Any]]) -> Any:
    for row in deliverables:
        if str(row.get("deliverable_id") or "") == "interactive_structure_browser_manifest":
            yield row


def _structure_browser_label(row: dict[str, Any]) -> str:
    label = str(row.get("display_label") or row.get("candidate_id") or "")
    rmsd = row.get("wt_runtime_ca_rmsd")
    plddt = row.get("plddt")
    if rmsd is not None and plddt is not None:
        return f"{label} | WT RMSD {float(rmsd):.2f} A | pLDDT {float(plddt):.1f}"
    return label


def _structure_browser_title(row: dict[str, Any]) -> str:
    label = str(row.get("display_label") or row.get("candidate_id") or "")
    seqid = row.get("sequence_identity_percent")
    rmsd = row.get("wt_runtime_ca_rmsd")
    details: list[str] = []
    if rmsd is not None:
        details.append(f"WT-runtime C-alpha RMSD {float(rmsd):.2f} A")
    if seqid is not None:
        details.append(f"sequence identity {float(seqid):.1f}%")
    return label if not details else f"{label} ({'; '.join(details)})"


def _structure_metric_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    fields = [
        "candidate_id",
        "group",
        "review_class",
        "review_rank",
        "plddt",
        "wt_runtime_ca_rmsd",
        "cryoem_mapped_ca_rmsd",
        "sequence_identity_percent",
        "mutation_count",
    ]
    return [{"field": field, "value": str(row.get(field) if row.get(field) is not None else "")} for field in fields]
