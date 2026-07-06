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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.pdb_alignment import (
    align_pdb_text_to_reference_ca,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    resolve_manifest_path,
)
from dnadesign.thread.structure_views import (
    StructureViewModel,
    StructureViewMoleculeStyle,
    StructureViewSelectionStyle,
    StructureViewSpec,
    render_structure_view_html,
    summarize_structure_atom_content,
)

from .notebook_structure_dashboard import (
    format_float,
    structure_browser_panel_html,
    structure_metric_rows,
    structure_summary_rows,
)
from .structure_browser_common import (
    DNA_CLASS_COLOR,
    PROTEIN_CLASS_COLOR,
    REFERENCE_COLOR,
    RESIDUE_CATEGORY_HIGHLIGHT_COLOR,
    RNA_CLASS_COLOR,
)

_STRUCTURE_VIEW_WIDTH = 760
_STRUCTURE_VIEW_HEIGHT = 520
_MOLECULE_CLASS_COLORS = {
    "protein": PROTEIN_CLASS_COLOR,
    "dna": DNA_CLASS_COLOR,
    "rna": RNA_CLASS_COLOR,
}


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
            enriched["_deliverable_description"] = str(manifest_row.get("description") or "")
            enriched["_deliverable_alt_text"] = str(manifest_row.get("alt_text") or "")
            enriched["_interpretation_limit"] = str(
                payload.get("interpretation_limit") or manifest_row.get("interpretation_limit") or ""
            )
            rows.append(enriched)
    return rows


def structure_browser_lookup(
    rows: list[dict[str, Any]],
    *,
    selected_section: str,
    selected_deliverable_id: str = "",
    selected_group: str = "",
) -> dict[str, dict[str, Any]]:
    """Build dropdown labels for the selected interactive structure deliverable."""

    return {
        _structure_browser_label(row): row
        for row in rows
        if str(row.get("_section") or "") == selected_section
        and str(row.get("_deliverable_id") or "") == selected_deliverable_id
        and (not selected_group or str(row.get("group") or "") == selected_group)
    }


def structure_group_lookup(
    rows: list[dict[str, Any]],
    *,
    selected_section: str,
    selected_deliverable_id: str = "",
) -> dict[str, str]:
    """Build structure group dropdown labels for the selected structure deliverable."""

    groups: dict[str, str] = {}
    for row in rows:
        if str(row.get("_section") or "") != selected_section:
            continue
        if str(row.get("_deliverable_id") or "") != selected_deliverable_id:
            continue
        group = str(row.get("group") or "Ungrouped structures")
        groups.setdefault(group, group)
    return groups


def structure_highlight_lookup(
    rows: list[dict[str, Any]],
    *,
    selected_row: dict[str, Any] | None,
) -> dict[str, dict[str, Any] | None]:
    """Build residue-highlight options for the currently selected structure."""

    if selected_row is None or str(selected_row.get("structure_view_mode") or "") == "reference_selection":
        return {}
    candidate_id = str(selected_row.get("source_candidate_id") or selected_row.get("candidate_id") or "")
    if not candidate_id:
        return {}
    options: dict[str, dict[str, Any] | None] = {"No residue highlight": None}
    for row in rows:
        if str(row.get("_deliverable_id") or "") != "mask_structure_browser_manifest":
            continue
        if str(row.get("structure_view_mode") or "") != "reference_selection":
            continue
        options[_structure_browser_label(row)] = row
    for row in rows:
        if str(row.get("_deliverable_id") or "") != "biohub_esmc_sae_structure_browser_manifest":
            continue
        if str(row.get("source_candidate_id") or row.get("candidate_id") or "") != candidate_id:
            continue
        if row.get("feature_index") is None:
            continue
        options[_structure_highlight_label(row)] = row
    return options


def render_structure_browser(
    *,
    mo: Any,
    selected_row: dict[str, Any] | None,
    structure_ui: Any,
    structure_group_ui: Any | None = None,
    structure_highlight_ui: Any | None = None,
    selected_highlight_row: dict[str, Any] | None = None,
    structure_background_ui: Any | None = None,
    structure_mutation_ui: Any | None = None,
    structure_sidechain_ui: Any | None = None,
    structure_protein_ui: Any | None = None,
    structure_dna_visible_ui: Any | None = None,
    structure_rna_visible_ui: Any | None = None,
    show_reference_background: bool = True,
    show_mutation_differences: bool = False,
    show_sidechains: bool = True,
    show_dna: bool = True,
    show_rna: bool = True,
    highlight_protein: bool = False,
    highlight_dna: bool = False,
    highlight_rna: bool = False,
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
    reference_format = str(reference.get("structure_format") or "pdb")
    query_format = str(selected_row.get("structure_format") or "pdb")
    reference_text = reference_path.read_text(encoding="utf-8")
    query_text = query_path.read_text(encoding="utf-8")
    reference_atom_content = summarize_structure_atom_content(reference_text, structure_format=reference_format)
    alignment_status = "raw_coordinates"
    browser_mapped_ca_rmsd: float | None = None
    alignment = dict(selected_row.get("_alignment") or {})
    view_mode = str(selected_row.get("structure_view_mode") or "")
    if view_mode == "reference_selection":
        query_text = ""
        alignment_status = "reference_selection"
    elif str(alignment.get("status") or "") == "enabled":
        alignment_reference_path = resolve_manifest_path(
            browser_root,
            str(alignment.get("reference_local_path") or reference.get("local_path") or ""),
        )
        if not alignment_reference_path.exists():
            return mo.md("Interactive structure alignment is skipped because the alignment reference PDB is missing.")
        alignment_reference_text = alignment_reference_path.read_text(encoding="utf-8")
        try:
            query_text, browser_mapped_ca_rmsd = align_pdb_text_to_reference_ca(
                query_text=query_text,
                reference_text=alignment_reference_text,
                query_start_residue=int(alignment.get("query_start_residue", 3)),
                reference_start_residue=int(alignment.get("reference_start_residue", 1)),
                residue_count=int(alignment.get("residue_count", 309)),
            )
            alignment_status = "aligned_in_memory_to_reference_ca"
        except Exception as exc:  # pragma: no cover - defensive notebook rendering path
            return mo.md(f"Interactive structure alignment failed: `{type(exc).__name__}: {exc}`")
    query_atom_content = (
        None if not query_text else summarize_structure_atom_content(query_text, structure_format=query_format)
    )
    query_model_id = str(selected_row.get("source_candidate_id") or selected_row["candidate_id"])
    selection_styles = _selection_styles(selected_row)
    selected_highlight_selection_styles = ()
    if selected_highlight_row is not None and selected_highlight_row is not selected_row:
        selected_highlight_selection_styles = _selection_styles(selected_highlight_row)
    mutation_selection_styles = ()
    if show_mutation_differences and view_mode != "reference_selection":
        mutation_selection_styles = _mutation_selection_styles(selected_row, model_id=query_model_id)
    residue_interest_overlay_active = bool(
        selection_styles or selected_highlight_selection_styles or mutation_selection_styles
    )
    reference_model = StructureViewModel(
        model_id=str(reference.get("model_id") or "reference"),
        structure_text=reference_text,
        structure_format=reference_format,
        label=str(reference.get("display_label") or "Reference"),
        color=str(reference.get("color") or "#d8d8d8"),
        opacity=0.82,
        show_sidechains=show_sidechains and reference_atom_content.has_sidechain_atoms,
    )
    models = []
    if view_mode == "reference_selection" or show_reference_background:
        models.append(reference_model)
    if view_mode != "reference_selection":
        models.append(
            StructureViewModel(
                model_id=query_model_id,
                structure_text=query_text,
                structure_format=query_format,
                label=str(selected_row.get("display_label") or selected_row["candidate_id"]),
                color=REFERENCE_COLOR
                if residue_interest_overlay_active
                else str(selected_row.get("color") or PROTEIN_CLASS_COLOR),
                show_sidechains=(
                    show_sidechains and query_atom_content.has_sidechain_atoms
                    if query_atom_content is not None
                    else False
                ),
            )
        )
    selection_styles += selected_highlight_selection_styles + mutation_selection_styles
    try:
        html_panel = render_structure_view_html(
            StructureViewSpec(
                title=_structure_browser_title(selected_row),
                subtitle=_structure_browser_subtitle(selected_row),
                description=_structure_browser_description(selected_row),
                interpretation_limit=_structure_browser_interpretation_limit(selected_row),
                models=tuple(models),
                molecule_styles=_molecule_styles(
                    models,
                    highlight_protein=highlight_protein,
                    highlight_dna=highlight_dna,
                    highlight_rna=highlight_rna,
                ),
                selection_styles=selection_styles,
                hidden_molecule_classes=_hidden_molecule_classes(show_dna=show_dna, show_rna=show_rna),
                width=_STRUCTURE_VIEW_WIDTH,
                height=_STRUCTURE_VIEW_HEIGHT,
                camera_memory_key=_camera_memory_key(selected_row),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive notebook rendering path
        return mo.md(f"Interactive structure viewer failed to render: `{type(exc).__name__}: {exc}`")
    metric_rows = structure_metric_rows(
        selected_row,
        selected_highlight_row=selected_highlight_row,
        alignment_status=alignment_status,
        browser_mapped_ca_rmsd=browser_mapped_ca_rmsd,
        reference_atom_content=reference_atom_content,
        query_atom_content=query_atom_content,
        show_sidechains=show_sidechains,
    )
    summary_rows = structure_summary_rows(
        selected_row,
        selected_highlight_row=selected_highlight_row,
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
    return mo.vstack(
        [
            mo.hstack(
                [
                    item
                    for item in (
                        structure_group_ui,
                        structure_ui,
                        structure_highlight_ui if view_mode != "reference_selection" else None,
                        structure_background_ui if view_mode != "reference_selection" else None,
                        structure_mutation_ui if view_mode != "reference_selection" else None,
                        structure_sidechain_ui,
                        structure_protein_ui,
                        structure_dna_visible_ui,
                        structure_rna_visible_ui,
                    )
                    if item is not None
                ],
                justify="center",
                align="stretch",
                wrap=True,
                gap=1.0,
                widths="equal",
            ),
            mo.Html(structure_browser_panel_html(html_panel, summary_rows, selected_row)),
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
        group = str(row.get("group") or "Reference selection")
        if residue_count is not None:
            return f"{group} | {int(residue_count)} highlighted residues"
        return group
    return str(row.get("group") or "Candidate fold overlay")


def _structure_browser_description(row: dict[str, Any]) -> str:
    row_description = str(row.get("description") or "").strip()
    if row_description:
        return row_description
    return str(row.get("_deliverable_description") or "").strip()


def _structure_browser_interpretation_limit(row: dict[str, Any]) -> str:
    return str(row.get("_interpretation_limit") or "").strip()


def _camera_memory_key(row: dict[str, Any]) -> str:
    deliverable_id = str(row.get("_deliverable_id") or "structure_browser")
    return f"eco1-rt-repack:{deliverable_id}:camera-v2"


def _molecule_styles(
    models: list[StructureViewModel],
    *,
    highlight_protein: bool,
    highlight_dna: bool,
    highlight_rna: bool,
) -> tuple[StructureViewMoleculeStyle, ...]:
    enabled = {
        "protein": highlight_protein,
        "dna": highlight_dna,
        "rna": highlight_rna,
    }
    labels = {
        "protein": "Protein",
        "dna": "DNA",
        "rna": "RNA",
    }
    styles: list[StructureViewMoleculeStyle] = []
    for model in models:
        for molecule_class, is_enabled in enabled.items():
            if not is_enabled:
                continue
            styles.append(
                StructureViewMoleculeStyle(
                    molecule_class=molecule_class,  # type: ignore[arg-type]
                    model_id=model.model_id,
                    label=labels[molecule_class],
                    color=_MOLECULE_CLASS_COLORS[molecule_class],
                )
            )
    return tuple(styles)


def _hidden_molecule_classes(*, show_dna: bool, show_rna: bool) -> tuple[str, ...]:
    hidden: list[str] = []
    if not show_dna:
        hidden.append("dna")
    if not show_rna:
        hidden.append("rna")
    return tuple(hidden)


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
                color=str(item.get("color") or RESIDUE_CATEGORY_HIGHLIGHT_COLOR),
                opacity=float(item.get("opacity", 1.0)),
                residue_scope=str(item.get("residue_scope") or "protein"),  # type: ignore[arg-type]
            )
        )
    return tuple(styles)


def _structure_highlight_label(row: dict[str, Any]) -> str:
    feature_index = int(row["feature_index"])
    rank_text = ""
    display_label = str(row.get("display_label") or "")
    if "peak order " in display_label:
        rank_text = display_label.split("peak order ", 1)[1].split("|", 1)[0].strip()
    activation_max = format_float(row.get("activation_max"), decimals=3)
    suffix = f" | max {activation_max}" if activation_max else ""
    if rank_text:
        return f"SAE F{feature_index} | peak order {rank_text}{suffix}"
    return f"SAE F{feature_index}{suffix}"


def _mutation_selection_styles(row: dict[str, Any], *, model_id: str) -> tuple[StructureViewSelectionStyle, ...]:
    residue_numbers = tuple(int(value) for value in row.get("mutation_residue_numbers") or [])
    if not residue_numbers:
        return ()
    return (
        StructureViewSelectionStyle(
            selection_id="candidate_differences",
            model_id=model_id,
            label="Candidate differences",
            residue_numbers=residue_numbers,
            color=RESIDUE_CATEGORY_HIGHLIGHT_COLOR,
        ),
    )
