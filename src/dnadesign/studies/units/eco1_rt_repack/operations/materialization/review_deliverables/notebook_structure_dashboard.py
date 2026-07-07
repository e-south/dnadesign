"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_structure_dashboard.py

Structure summary helpers for Eco1 review-notebook structure views.

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
        "c_terminal_primer_rna_recognition_mutation_count",
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
        rows.extend(_selected_highlight_metric_rows(selected_highlight_row))
        for field in _selected_sae_metric_fields(selected_highlight_row):
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


def structure_summary_rows(
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
    summary: list[dict[str, str]] = [
        {"metric": "Structure", "value": str(row.get("display_label") or row.get("candidate_id") or "")},
        {"metric": "Fold-review bin", "value": str(row.get("group") or "")},
    ]
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        summary.extend(_reference_summary_rows(row))
    else:
        summary.extend(_candidate_summary_rows(row))
    if selected_highlight_row is not None:
        summary.extend(_selected_highlight_summary_rows(selected_highlight_row))
    summary.extend(
        _browser_summary_rows(
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
    return [entry for entry in summary if str(entry.get("value") or "")]


def structure_browser_panel_html(
    structure_html: str,
    summary_rows: list[dict[str, str]],
    row: dict[str, Any],
) -> str:
    summary_html = _summary_table_html(row, summary_rows)
    sequence_html = _protein_sequence_panel_html(row)
    return (
        '<section class="eco1-structure-browser-panel">'
        "<style>"
        ".eco1-structure-browser-panel{width:100%;max-width:100%;min-width:0;box-sizing:border-box;}"
        ".eco1-structure-browser-grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(14rem,0.82fr);"
        "gap:0.75rem;align-items:start;width:100%;max-width:100%;min-width:0;box-sizing:border-box;}"
        ".eco1-structure-browser-view{min-width:0;max-width:100%;overflow:hidden;}"
        ".eco1-structure-browser-summary{min-width:0;max-width:100%;overflow:hidden;}"
        ".eco1-structure-browser-summary table{width:100%;border-collapse:collapse;table-layout:fixed;"
        "font-size:0.86rem;line-height:1.25;}"
        ".eco1-structure-browser-summary th{font-size:0.78rem;text-transform:uppercase;"
        "letter-spacing:0.04em;color:#6e7781;text-align:left;padding:0 0 0.28rem 0;}"
        ".eco1-structure-browser-summary td{border-top:1px solid #d8dee4;padding:0.34rem 0.25rem;"
        "vertical-align:top;overflow-wrap:anywhere;}"
        ".eco1-structure-browser-summary td:first-child{width:38%;color:#57606a;font-weight:600;}"
        ".eco1-protein-sequence-panel{border-top:1px solid #d8dee4;margin-top:0.55rem;padding-top:0.45rem;}"
        ".eco1-protein-sequence-panel h4{margin:0 0 0.28rem 0;font-size:0.78rem;text-transform:uppercase;"
        "letter-spacing:0.04em;color:#6e7781;}"
        ".eco1-protein-sequence{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
        "font-size:0.76rem;line-height:1.35;white-space:normal;overflow-wrap:anywhere;word-break:break-word;}"
        "@media (max-width: 860px){.eco1-structure-browser-grid{grid-template-columns:1fr;}}"
        "</style>"
        '<div class="eco1-structure-browser-grid">'
        f'<div class="eco1-structure-browser-view">{structure_html}</div>'
        f'<aside class="eco1-structure-browser-summary">{summary_html}{sequence_html}</aside>'
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


def _candidate_summary_rows(row: dict[str, Any]) -> list[dict[str, str]]:
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
        {"metric": "Protein sequence length", "value": format_int(row.get("protein_sequence_length"))},
        {"metric": "Mutation count", "value": format_int(row.get("mutation_count"))},
        {"metric": "Panel slot", "value": str(row.get("selection_slot") or "")},
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
            "metric": "Near retained DNA/RNA edits",
            "value": format_int(row.get("nucleic_acid_facing_mutation_count")),
        },
        {
            "metric": "Near-region charge change",
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
        {
            "metric": "C-terminal primer-RNA recognition changes",
            "value": format_int(row.get("c_terminal_primer_rna_recognition_mutation_count")),
        },
        {"metric": "Distal scaffold changes", "value": format_int(row.get("distal_scaffold_mutation_count"))},
    ]


def _reference_summary_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"metric": "Highlighted residues", "value": format_int(row.get("selection_residue_count"))},
        {"metric": "Evidence view", "value": str(row.get("group") or "Reference evidence")},
        {"metric": "Base model", "value": "Ec86/7V9U"},
        {"metric": "Candidate structure", "value": "No candidate structure is shown in this view."},
    ]


def _selected_highlight_metric_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    label = str(row.get("display_label") or row.get("candidate_id") or "")
    return [
        {"field": "selected_residue_highlight", "value": label},
        {"field": "selected_residue_highlight_group", "value": str(row.get("group") or "")},
        {"field": "selected_residue_highlight_count", "value": format_metric_value(row.get("selection_residue_count"))},
    ]


def _selected_sae_metric_fields(row: dict[str, Any]) -> tuple[str, ...]:
    if row.get("feature_index") is None:
        return ()
    return ("feature_index", "activation_max", "activation_sum", "nonzero_residue_count")


def _selected_highlight_summary_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    if row.get("feature_index") is not None:
        return _selected_sae_summary_rows(row)
    label = str(row.get("display_label") or row.get("candidate_id") or "")
    return [
        {"metric": "Selected residue highlight", "value": label},
        {"metric": "Highlight category", "value": str(row.get("group") or "")},
        {"metric": "Highlighted residues", "value": format_int(row.get("selection_residue_count"))},
    ]


def _selected_sae_summary_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    feature_value = f"F{int(row['feature_index'])}" if row.get("feature_index") is not None else ""
    return [
        {"metric": "Selected SAE feature", "value": feature_value},
        {"metric": "SAE activation max", "value": format_float(row.get("activation_max"), decimals=3)},
        {"metric": "SAE activation sum", "value": format_float(row.get("activation_sum"), decimals=3)},
        {"metric": "SAE nonzero residues", "value": format_int(row.get("nonzero_residue_count"))},
    ]


def _browser_summary_rows(
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
        {"metric": "Reference atoms", "value": _atom_content_summary_value(reference_atom_content)},
        {
            "metric": "Query atoms",
            "value": "" if query_atom_content is None else _atom_content_summary_value(query_atom_content),
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
            "value": molecule_visibility_summary_value(show_dna=show_dna, show_rna=show_rna),
        },
        {
            "metric": "Molecule colors",
            "value": molecule_color_summary_value(
                highlight_protein=highlight_protein,
                highlight_dna=highlight_dna,
                highlight_rna=highlight_rna,
            ),
        },
    ]


def _summary_table_html(row: dict[str, Any], summary_rows: list[dict[str, str]]) -> str:
    label = (
        "Reference summary" if str(row.get("structure_view_mode") or "") == "reference_selection" else "Variant summary"
    )
    body = "".join(_summary_row_html(entry) for entry in summary_rows)
    return (
        f'<table aria-label="{html.escape(label)}">'
        f'<thead><tr><th colspan="2">{html.escape(label)}</th></tr></thead>'
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def _summary_row_html(entry: dict[str, str]) -> str:
    metric = str(entry.get("metric") or "")
    value = str(entry.get("value") or "")
    value_html = html.escape(value)
    return f"<tr><td>{html.escape(metric)}</td><td>{value_html}</td></tr>"


def _protein_sequence_panel_html(row: dict[str, Any]) -> str:
    sequence = str(row.get("protein_sequence") or "").strip()
    if not sequence:
        return ""
    label = "Protein sequence"
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        label = "Reference protein sequence"
    return (
        '<section class="eco1-protein-sequence-panel">'
        f"<h4>{html.escape(label)}</h4>"
        f'<code class="eco1-protein-sequence">{html.escape(sequence)}</code>'
        "</section>"
    )


def _atom_content_summary_value(content: StructureAtomContent) -> str:
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


def molecule_color_summary_value(
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


def molecule_visibility_summary_value(*, show_dna: bool, show_rna: bool) -> str:
    visible = ["Protein"]
    if show_dna:
        visible.append("DNA")
    if show_rna:
        visible.append("RNA")
    return ", ".join(visible) + "."
