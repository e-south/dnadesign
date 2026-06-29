"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_structure_browser.py

Interactive structure-browser helpers for the Eco1 review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.pdb_alignment import (
    align_pdb_text_to_reference_ca,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    resolve_manifest_path,
)
from dnadesign.thread.structure_views import (
    StructureViewModel,
    StructureViewSelectionStyle,
    StructureViewSpec,
    render_structure_view_html,
)

_STRUCTURE_VIEW_SIZE = 640


def load_structure_browser_rows(*, manifest_root: Path, deliverables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load interactive structure-browser rows if the manifest is materialized."""

    rows: list[dict[str, Any]] = []
    for manifest_row in _interactive_structure_manifest_rows(deliverables):
        if str(manifest_row.get("status") or "") != "rendered":
            continue
        manifest_path = resolve_manifest_path(manifest_root, str(manifest_row["path"]))
        if not manifest_path.exists():
            continue
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        browser_root = manifest_path.parent
        reference = dict(payload.get("reference") or {})
        alignment = dict(payload.get("alignment") or {})
        for row in payload.get("structures") or []:
            if not isinstance(row, dict):
                continue
            enriched = dict(row)
            enriched["_browser_root"] = str(browser_root)
            enriched["_reference"] = reference
            enriched["_alignment"] = alignment
            enriched["_deliverable_id"] = str(manifest_row.get("deliverable_id") or "")
            enriched["_section"] = str(manifest_row.get("section") or "")
            enriched["_control_label"] = str(payload.get("control_label") or "Structure view")
            rows.append(enriched)
    return rows


def structure_browser_lookup(
    rows: list[dict[str, Any]],
    *,
    selected_section: str,
    selected_deliverable_id: str = "",
) -> dict[str, dict[str, Any]]:
    """Build dropdown labels for the selected interactive structure deliverable."""

    return {
        _structure_browser_label(row): row
        for row in rows
        if str(row.get("_section") or "") == selected_section
        and str(row.get("_deliverable_id") or "") == selected_deliverable_id
    }


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
    reference_text = reference_path.read_text(encoding="utf-8")
    query_text = query_path.read_text(encoding="utf-8")
    alignment_status = "raw_coordinates"
    browser_mapped_ca_rmsd: float | None = None
    alignment = dict(selected_row.get("_alignment") or {})
    view_mode = str(selected_row.get("structure_view_mode") or "")
    if view_mode == "reference_selection":
        query_text = ""
        alignment_status = "reference_selection"
    elif str(alignment.get("status") or "") == "enabled":
        try:
            query_text, browser_mapped_ca_rmsd = align_pdb_text_to_reference_ca(
                query_text=query_text,
                reference_text=reference_text,
                query_start_residue=int(alignment.get("query_start_residue", 3)),
                reference_start_residue=int(alignment.get("reference_start_residue", 1)),
                residue_count=int(alignment.get("residue_count", 309)),
            )
            alignment_status = "aligned_in_memory_to_reference_ca"
        except Exception as exc:  # pragma: no cover - defensive notebook rendering path
            return mo.md(f"Interactive structure alignment failed: `{type(exc).__name__}: {exc}`")
    models = [
        StructureViewModel(
            model_id=str(reference.get("model_id") or "reference"),
            structure_text=reference_text,
            label=str(reference.get("display_label") or "Reference"),
            color=str(reference.get("color") or "#d8d8d8"),
            opacity=0.82,
        )
    ]
    if view_mode != "reference_selection":
        models.append(
            StructureViewModel(
                model_id=str(selected_row["candidate_id"]),
                structure_text=query_text,
                label=str(selected_row.get("display_label") or selected_row["candidate_id"]),
                color=str(selected_row.get("color") or "#0072B2"),
            )
        )
    try:
        html_panel = render_structure_view_html(
            StructureViewSpec(
                title=_structure_browser_title(selected_row),
                subtitle=_structure_browser_subtitle(selected_row),
                models=tuple(models),
                selection_styles=_selection_styles(selected_row),
                width=_STRUCTURE_VIEW_SIZE,
                height=_STRUCTURE_VIEW_SIZE,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive notebook rendering path
        return mo.md(f"Interactive structure viewer failed to render: `{type(exc).__name__}: {exc}`")
    metric_rows = _structure_metric_rows(
        selected_row,
        alignment_status=alignment_status,
        browser_mapped_ca_rmsd=browser_mapped_ca_rmsd,
    )
    return mo.vstack(
        [
            mo.hstack([structure_ui], justify="center", gap=1.0),
            mo.Html(html_panel),
            mo.Html(_structure_metric_summary(selected_row)),
            mo.Html(_alignment_note(alignment_status, browser_mapped_ca_rmsd)),
            mo.accordion(
                {"Selected structure evidence": mo.ui.table(metric_rows, page_size=8)},
                multiple=False,
                lazy=True,
            ),
        ],
        gap=0.35,
    )


def _interactive_structure_manifest_rows(deliverables: list[dict[str, Any]]) -> Any:
    for row in deliverables:
        if str(row.get("artifact_kind") or "") == "structure_browser_manifest":
            yield row


def _structure_browser_label(row: dict[str, Any]) -> str:
    label = str(row.get("display_label") or row.get("candidate_id") or "")
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        residue_count = row.get("selection_residue_count")
        if residue_count is not None:
            return f"{label} | {int(residue_count)} residues"
        return label
    rmsd = row.get("wt_runtime_ca_rmsd")
    plddt = row.get("plddt")
    if rmsd is not None and plddt is not None:
        return f"{label} | WT RMSD {float(rmsd):.2f} A | pLDDT {float(plddt):.1f}"
    return label


def _structure_browser_title(row: dict[str, Any]) -> str:
    label = str(row.get("display_label") or row.get("candidate_id") or "")
    return label


def _structure_browser_subtitle(row: dict[str, Any]) -> str:
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        residue_count = row.get("selection_residue_count")
        if residue_count is not None:
            return f"Reference mask evidence | {int(residue_count)} highlighted residues"
        return "Reference mask evidence"
    seqid = row.get("sequence_identity_percent")
    rmsd = row.get("wt_runtime_ca_rmsd")
    cryoem_rmsd = row.get("cryoem_mapped_ca_rmsd")
    plddt = row.get("plddt")
    mutation_count = row.get("mutation_count")
    details: list[str] = []
    if plddt is not None:
        details.append(f"mean pLDDT {float(plddt):.1f}")
    if rmsd is not None:
        details.append(f"WT-runtime C-alpha RMSD {float(rmsd):.2f} A")
    if cryoem_rmsd is not None:
        details.append(f"direct cryoEM C-alpha RMSD {float(cryoem_rmsd):.2f} A")
    if seqid is not None:
        details.append(f"sequence identity {float(seqid):.1f}%")
    if mutation_count is not None:
        details.append(f"{int(mutation_count)} mutations")
    return " | ".join(details)


def _structure_metric_summary(row: dict[str, Any]) -> str:
    cards = (
        _reference_metric_cards(row)
        if str(row.get("structure_view_mode") or "") == "reference_selection"
        else [
            ("Mean pLDDT", _format_float(row.get("plddt"), decimals=1)),
            ("WT-runtime CA RMSD", _format_float(row.get("wt_runtime_ca_rmsd"), decimals=2, suffix=" A")),
            ("Direct cryoEM CA RMSD", _format_float(row.get("cryoem_mapped_ca_rmsd"), decimals=2, suffix=" A")),
            ("Sequence identity", _format_float(row.get("sequence_identity_percent"), decimals=1, suffix="%")),
            ("Mutations", _format_int(row.get("mutation_count"))),
        ]
    )
    card_html = "".join(_metric_card(label, value) for label, value in cards if value)
    if not card_html:
        return ""
    return (
        '<section aria-label="Structure metric summary" '
        'style="margin:0.1rem auto 0 auto; max-width:672px; width:min(100%, 672px);">'
        '<div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em; '
        'color:#6e7781; text-align:center; margin-bottom:0.25rem;">Structure metric summary</div>'
        '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(8.2rem, 1fr)); '
        'gap:0.35rem;">'
        f"{card_html}"
        "</div></section>"
    )


def _reference_metric_cards(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Highlighted residues", _format_int(row.get("selection_residue_count"))),
        ("Evidence view", html.escape(str(row.get("group") or "Reference mask evidence"))),
        ("Base model", "ec86kit/7V9U"),
    ]


def _metric_card(label: str, value: str) -> str:
    return (
        '<div style="border:1px solid #d8dee4; border-radius:6px; background:#ffffff; '
        'padding:0.42rem 0.5rem; text-align:center;">'
        f'<div style="font-size:0.72rem; color:#6e7781; line-height:1.18;">{html.escape(label)}</div>'
        f'<div style="font-size:0.98rem; font-weight:650; color:#24292f; line-height:1.25;">'
        f"{html.escape(value)}</div>"
        "</div>"
    )


def _format_float(value: Any, *, decimals: int, suffix: str = "") -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return ""


def _format_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def _alignment_note(alignment_status: str, browser_mapped_ca_rmsd: float | None) -> str:
    if alignment_status == "reference_selection":
        return (
            '<div style="border-left:3px solid #0969da; padding:0.35rem 0.55rem; '
            'background:#f6f8fa; color:#24292f; font-size:0.9rem; line-height:1.35;">'
            "<strong>Reference selection:</strong> highlighted residues come from the current "
            "Eco1 mask evidence. No candidate structure is shown in this view."
            "</div>"
        )
    rmsd_text = (
        "" if browser_mapped_ca_rmsd is None else f" Browser mapped C-alpha RMSD: {browser_mapped_ca_rmsd:.3f} A."
    )
    return (
        '<div style="border-left:3px solid #0969da; padding:0.35rem 0.55rem; '
        'background:#f6f8fa; color:#24292f; font-size:0.9rem; line-height:1.35;">'
        f"<strong>Browser alignment:</strong> {alignment_status}.{rmsd_text} "
        "Raw local ColabFold PDB files are not rewritten."
        "</div>"
    )


def _structure_metric_rows(
    row: dict[str, Any],
    *,
    alignment_status: str,
    browser_mapped_ca_rmsd: float | None,
) -> list[dict[str, str]]:
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
    rows = [{"field": field, "value": str(row.get(field) if row.get(field) is not None else "")} for field in fields]
    rows.extend(
        [
            {"field": "browser_alignment_status", "value": alignment_status},
            {
                "field": "browser_mapped_ca_rmsd",
                "value": "" if browser_mapped_ca_rmsd is None else f"{browser_mapped_ca_rmsd:.3f}",
            },
        ]
    )
    return rows


def _selection_styles(row: dict[str, Any]) -> tuple[StructureViewSelectionStyle, ...]:
    styles: list[StructureViewSelectionStyle] = []
    for item in row.get("selection_styles") or []:
        if not isinstance(item, dict):
            continue
        styles.append(
            StructureViewSelectionStyle(
                selection_id=str(item.get("selection_id") or ""),
                model_id=str(item.get("model_id") or ""),
                label=str(item.get("label") or ""),
                residue_numbers=tuple(int(value) for value in item.get("residue_numbers") or []),
                color=str(item.get("color") or "#D55E00"),
                opacity=float(item.get("opacity", 1.0)),
            )
        )
    return tuple(styles)
