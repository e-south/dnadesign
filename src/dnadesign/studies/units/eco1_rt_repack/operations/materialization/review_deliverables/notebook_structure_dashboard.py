"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_structure_dashboard.py

Dashboard helpers for Eco1 review-notebook structure views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
from typing import Any

from dnadesign.thread.structure_views import StructureAtomContent


def structure_metric_rows(
    row: dict[str, Any],
    *,
    selected_highlight_row: dict[str, Any] | None,
    alignment_status: str,
    browser_mapped_ca_rmsd: float | None,
    reference_atom_content: StructureAtomContent,
    query_atom_content: StructureAtomContent | None,
    show_sidechains: bool,
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
        "canonical_mutations",
        "mutation_residue_numbers",
        "selection_slot",
        "nearest_selected_distance_aa",
        "selection_support_alt_observed_fraction",
        "selection_support_unobserved_mutation_count",
        "nucleic_acid_facing_mutation_count",
        "nucleic_acid_facing_charge_delta",
        "nucleic_acid_facing_chemistry_warning_count",
        "catalytic_or_direct_contact_mutation_count",
        "thumb_contact_track_mutation_count",
        "distal_scaffold_mutation_count",
        "feature_index",
        "activation_max",
        "activation_sum",
        "nonzero_residue_count",
        "esmc_llr_total",
        "esmc_llr_per_mutation",
        "esmc_mutations_scored_count",
        "esmc_scoring_method_id",
    ]
    rows = [{"field": field, "value": format_metric_value(row.get(field))} for field in fields]
    if selected_highlight_row is not None:
        for field in ("feature_index", "activation_max", "activation_sum", "nonzero_residue_count"):
            rows.append(
                {
                    "field": f"selected_sae_{field}",
                    "value": format_metric_value(selected_highlight_row.get(field)),
                }
            )
    rows.extend(
        [
            {"field": "browser_alignment_status", "value": alignment_status},
            {
                "field": "browser_mapped_ca_rmsd",
                "value": "" if browser_mapped_ca_rmsd is None else f"{browser_mapped_ca_rmsd:.3f}",
            },
            {"field": "reference_atom_scope", "value": reference_atom_content.scope_label},
            {"field": "reference_atom_count", "value": str(reference_atom_content.atom_count)},
            {
                "field": "reference_sidechain_residue_count",
                "value": str(reference_atom_content.sidechain_residue_count),
            },
            {
                "field": "query_atom_scope",
                "value": "" if query_atom_content is None else query_atom_content.scope_label,
            },
            {
                "field": "query_atom_count",
                "value": "" if query_atom_content is None else str(query_atom_content.atom_count),
            },
            {
                "field": "query_sidechain_residue_count",
                "value": "" if query_atom_content is None else str(query_atom_content.sidechain_residue_count),
            },
            {
                "field": "sidechain_display",
                "value": sidechain_display_message(
                    row,
                    reference_atom_content=reference_atom_content,
                    query_atom_content=query_atom_content,
                    show_sidechains=show_sidechains,
                ),
            },
            {
                "field": "browser_alignment_note",
                "value": browser_alignment_note(
                    row,
                    alignment_status=alignment_status,
                    browser_mapped_ca_rmsd=browser_mapped_ca_rmsd,
                ),
            },
        ]
    )
    return rows


def structure_dashboard_rows(
    row: dict[str, Any],
    *,
    selected_highlight_row: dict[str, Any] | None,
    alignment_status: str,
    browser_mapped_ca_rmsd: float | None,
    reference_atom_content: StructureAtomContent,
    query_atom_content: StructureAtomContent | None,
    show_sidechains: bool,
    show_dna: bool,
    show_rna: bool,
    highlight_protein: bool,
    highlight_dna: bool,
    highlight_rna: bool,
) -> list[dict[str, str]]:
    dashboard: list[dict[str, str]] = [
        {"metric": "Structure", "value": str(row.get("display_label") or row.get("candidate_id") or "")},
        {"metric": "Fold-review bin", "value": str(row.get("group") or "")},
    ]
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        dashboard.extend(_reference_dashboard_rows(row))
    else:
        dashboard.extend(_candidate_dashboard_rows(row))
    if selected_highlight_row is not None:
        dashboard.extend(_selected_sae_dashboard_rows(selected_highlight_row))
    dashboard.extend(
        _browser_dashboard_rows(
            row,
            alignment_status=alignment_status,
            browser_mapped_ca_rmsd=browser_mapped_ca_rmsd,
            reference_atom_content=reference_atom_content,
            query_atom_content=query_atom_content,
            show_sidechains=show_sidechains,
            show_dna=show_dna,
            show_rna=show_rna,
            highlight_protein=highlight_protein,
            highlight_dna=highlight_dna,
            highlight_rna=highlight_rna,
        )
    )
    return [entry for entry in dashboard if str(entry.get("value") or "")]


def structure_browser_panel_html(
    structure_html: str,
    dashboard_rows: list[dict[str, str]],
    row: dict[str, Any],
) -> str:
    return (
        '<section class="eco1-structure-browser-panel">'
        "<style>"
        ".eco1-structure-browser-panel{width:100%;max-width:100%;min-width:0;box-sizing:border-box;}"
        ".eco1-structure-browser-grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(14rem,0.82fr);"
        "gap:0.75rem;align-items:start;width:100%;max-width:100%;min-width:0;box-sizing:border-box;}"
        ".eco1-structure-browser-view{min-width:0;max-width:100%;overflow:hidden;}"
        ".eco1-structure-browser-dashboard{min-width:0;max-width:100%;overflow:hidden;}"
        ".eco1-structure-browser-dashboard table{width:100%;border-collapse:collapse;table-layout:fixed;"
        "font-size:0.86rem;line-height:1.25;}"
        ".eco1-structure-browser-dashboard th{font-size:0.78rem;text-transform:uppercase;"
        "letter-spacing:0.04em;color:#6e7781;text-align:left;padding:0 0 0.28rem 0;}"
        ".eco1-structure-browser-dashboard td{border-top:1px solid #d8dee4;padding:0.34rem 0.25rem;"
        "vertical-align:top;overflow-wrap:anywhere;}"
        ".eco1-structure-browser-dashboard td:first-child{width:38%;color:#57606a;font-weight:600;}"
        "@media (max-width: 860px){.eco1-structure-browser-grid{grid-template-columns:1fr;}}"
        "</style>"
        '<div class="eco1-structure-browser-grid">'
        f'<div class="eco1-structure-browser-view">{structure_html}</div>'
        f'<aside class="eco1-structure-browser-dashboard">{_dashboard_table_html(row, dashboard_rows)}</aside>'
        "</div>"
        "</section>"
    )


def format_float(value: Any, *, decimals: int, suffix: str = "") -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return ""


def format_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def format_metric_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _candidate_dashboard_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"metric": "Mean pLDDT", "value": format_float(row.get("plddt"), decimals=1)},
        {"metric": "WT-runtime CA RMSD", "value": format_float(row.get("wt_runtime_ca_rmsd"), decimals=2, suffix=" A")},
        {
            "metric": "Direct cryoEM CA RMSD",
            "value": format_float(row.get("cryoem_mapped_ca_rmsd"), decimals=2, suffix=" A"),
        },
        {
            "metric": "Sequence identity",
            "value": format_float(row.get("sequence_identity_percent"), decimals=1, suffix="%"),
        },
        {"metric": "Mutation count", "value": format_int(row.get("mutation_count"))},
        {"metric": "Selection slot", "value": str(row.get("selection_slot") or "")},
        {"metric": "Nearest selected distance", "value": format_int(row.get("nearest_selected_distance_aa"))},
        {
            "metric": "MSA observed fraction",
            "value": format_float(row.get("selection_support_alt_observed_fraction"), decimals=3),
        },
        {
            "metric": "Unobserved MSA changes",
            "value": format_int(row.get("selection_support_unobserved_mutation_count")),
        },
        {
            "metric": "NA-facing mutations",
            "value": format_int(row.get("nucleic_acid_facing_mutation_count")),
        },
        {
            "metric": "NA-facing charge change",
            "value": format_int(row.get("nucleic_acid_facing_charge_delta")),
        },
        {
            "metric": "Chemistry warnings",
            "value": format_int(row.get("nucleic_acid_facing_chemistry_warning_count")),
        },
        {
            "metric": "Catalytic/direct-contact changes",
            "value": format_int(row.get("catalytic_or_direct_contact_mutation_count")),
        },
        {
            "metric": "Thumb contact-track changes",
            "value": format_int(row.get("thumb_contact_track_mutation_count")),
        },
        {"metric": "Distal scaffold changes", "value": format_int(row.get("distal_scaffold_mutation_count"))},
        {"metric": "ESMC additive LLR total", "value": format_float(row.get("esmc_llr_total"), decimals=2)},
        {"metric": "ESMC additive LLR / mutation", "value": format_float(row.get("esmc_llr_per_mutation"), decimals=2)},
        {"metric": "ESMC scoring method", "value": str(row.get("esmc_scoring_method_id") or "")},
    ]


def _reference_dashboard_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"metric": "Highlighted residues", "value": format_int(row.get("selection_residue_count"))},
        {"metric": "Evidence view", "value": str(row.get("group") or "Reference mask evidence")},
        {"metric": "Base model", "value": "Ec86/7V9U"},
        {"metric": "Candidate structure", "value": "No candidate structure is shown in this view."},
    ]


def _selected_sae_dashboard_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    feature_value = f"F{int(row['feature_index'])}" if row.get("feature_index") is not None else ""
    return [
        {"metric": "Selected SAE feature", "value": feature_value},
        {"metric": "SAE activation max", "value": format_float(row.get("activation_max"), decimals=3)},
        {"metric": "SAE activation sum", "value": format_float(row.get("activation_sum"), decimals=3)},
        {"metric": "SAE nonzero residues", "value": format_int(row.get("nonzero_residue_count"))},
    ]


def _browser_dashboard_rows(
    row: dict[str, Any],
    *,
    alignment_status: str,
    browser_mapped_ca_rmsd: float | None,
    reference_atom_content: StructureAtomContent,
    query_atom_content: StructureAtomContent | None,
    show_sidechains: bool,
    show_dna: bool,
    show_rna: bool,
    highlight_protein: bool,
    highlight_dna: bool,
    highlight_rna: bool,
) -> list[dict[str, str]]:
    return [
        {"metric": "Browser alignment", "value": alignment_status},
        {
            "metric": "Browser mapped CA RMSD",
            "value": "" if browser_mapped_ca_rmsd is None else f"{browser_mapped_ca_rmsd:.3f} A",
        },
        {"metric": "Reference atoms", "value": _atom_content_dashboard_value(reference_atom_content)},
        {
            "metric": "Query atoms",
            "value": "" if query_atom_content is None else _atom_content_dashboard_value(query_atom_content),
        },
        {
            "metric": "Side chains",
            "value": sidechain_display_message(
                row,
                reference_atom_content=reference_atom_content,
                query_atom_content=query_atom_content,
                show_sidechains=show_sidechains,
            ),
        },
        {
            "metric": "Molecule visibility",
            "value": molecule_visibility_dashboard_value(show_dna=show_dna, show_rna=show_rna),
        },
        {
            "metric": "Molecule colors",
            "value": molecule_color_dashboard_value(
                highlight_protein=highlight_protein,
                highlight_dna=highlight_dna,
                highlight_rna=highlight_rna,
            ),
        },
    ]


def _dashboard_table_html(row: dict[str, Any], dashboard_rows: list[dict[str, str]]) -> str:
    label = (
        "Reference dashboard"
        if str(row.get("structure_view_mode") or "") == "reference_selection"
        else "Variant dashboard"
    )
    body = "".join(
        "<tr>"
        f"<td>{html.escape(str(entry.get('metric') or ''))}</td>"
        f"<td>{html.escape(str(entry.get('value') or ''))}</td>"
        "</tr>"
        for entry in dashboard_rows
    )
    return (
        f'<table aria-label="{html.escape(label)}">'
        f'<thead><tr><th colspan="2">{html.escape(label)}</th></tr></thead>'
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def _atom_content_dashboard_value(content: StructureAtomContent) -> str:
    if content.has_sidechain_atoms:
        return f"{content.atom_count} protein atoms; {content.sidechain_residue_count} residues with side-chain atoms"
    return f"{content.atom_count} protein atoms; no side-chain atoms detected"


def sidechain_display_message(
    row: dict[str, Any],
    *,
    reference_atom_content: StructureAtomContent,
    query_atom_content: StructureAtomContent | None,
    show_sidechains: bool,
) -> str:
    if not show_sidechains:
        return "Side-chain sticks are toggled off for this view."
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        if reference_atom_content.has_sidechain_atoms:
            return "Reference side-chain atoms are present and rendered as sticks."
        return "The selected reference file is backbone-only, so side-chain sticks cannot be rendered for this view."
    query_message = (
        "Candidate side-chain atoms are present and rendered as sticks."
        if query_atom_content is not None and query_atom_content.has_sidechain_atoms
        else "The selected candidate PDB has no side-chain atoms, so candidate sticks are not rendered."
    )
    reference_message = (
        "The reference background includes protein side-chain atoms rendered as sticks."
        if reference_atom_content.has_sidechain_atoms
        else "The reference background is backbone-only."
    )
    return f"{query_message} {reference_message}"


def browser_alignment_note(
    row: dict[str, Any],
    *,
    alignment_status: str,
    browser_mapped_ca_rmsd: float | None,
) -> str:
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        return "No candidate structure is shown in this view."
    rmsd_text = (
        "" if browser_mapped_ca_rmsd is None else f" Browser mapped C-alpha RMSD: {browser_mapped_ca_rmsd:.3f} A."
    )
    return f"{alignment_status}.{rmsd_text} Raw local ColabFold PDB files are not rewritten."


def molecule_color_dashboard_value(
    *,
    highlight_protein: bool,
    highlight_dna: bool,
    highlight_rna: bool,
) -> str:
    enabled = [
        label
        for label, is_enabled in (
            ("Protein", highlight_protein),
            ("DNA", highlight_dna),
            ("RNA", highlight_rna),
        )
        if is_enabled
    ]
    if not enabled:
        return "Off."
    return "On for " + ", ".join(enabled) + "."


def molecule_visibility_dashboard_value(*, show_dna: bool, show_rna: bool) -> str:
    visible = ["Protein"]
    if show_dna:
        visible.append("DNA")
    if show_rna:
        visible.append("RNA")
    return ", ".join(visible) + "."
