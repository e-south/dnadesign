"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook.py

Marimo notebook writer for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def write_review_deliverables_notebook(path: Path) -> None:
    """Write a compact marimo notebook for Eco1 review deliverables."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_NOTEBOOK_SOURCE, encoding="utf-8")


_NOTEBOOK_SOURCE = """import marimo

__generated_with = "dnadesign.eco1_rt_repack.review_deliverables"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
        deliverable_lookup, format_deliverable_label, is_interactive_structure_deliverable, load_review_manifest,
        render_deliverable_details, render_deliverable_panel, render_intro, review_lane_lookup, section_deliverables,
        section_label_lookup, selected_deliverable, visual_deliverables,
    )
    from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
        notebook_sae_features as sae_runtime,
        notebook_structure_browser as structure_runtime,
    )

    is_sae_feature_heatmap_deliverable = sae_runtime.is_sae_feature_heatmap_deliverable
    load_sae_feature_heatmap_manifest = sae_runtime.load_sae_feature_heatmap_manifest
    load_structure_browser_rows = structure_runtime.load_structure_browser_rows
    render_sae_feature_heatmap = sae_runtime.render_sae_feature_heatmap
    sae_heatmap_feature_lookup = sae_runtime.sae_heatmap_feature_lookup
    render_structure_browser = structure_runtime.render_structure_browser
    structure_browser_lookup = structure_runtime.structure_browser_lookup
    structure_group_lookup = structure_runtime.structure_group_lookup
    structure_highlight_lookup = structure_runtime.structure_highlight_lookup

    return (
        deliverable_lookup, format_deliverable_label, is_interactive_structure_deliverable,
        is_sae_feature_heatmap_deliverable, load_review_manifest, load_sae_feature_heatmap_manifest,
        load_structure_browser_rows, mo, render_deliverable_details, render_deliverable_panel, render_intro,
        render_sae_feature_heatmap, render_structure_browser, review_lane_lookup, sae_heatmap_feature_lookup,
        section_deliverables, section_label_lookup, selected_deliverable, structure_browser_lookup,
        structure_group_lookup, structure_highlight_lookup, visual_deliverables,
    )


@app.cell
def _(load_review_manifest):
    manifest, deliverables, manifest_path, manifest_root = load_review_manifest(__file__)
    return deliverables, manifest, manifest_path, manifest_root


@app.cell
def _(mo, render_intro):
    render_intro(mo)


@app.cell
def _(deliverables, review_lane_lookup):
    lane_lookup = review_lane_lookup(deliverables)
    lane_options = list(lane_lookup)
    return lane_lookup, lane_options


@app.cell
def _(lane_options, mo):
    review_lane_ui = mo.ui.dropdown(
        lane_options,
        value=lane_options[0] if lane_options else None,
        label="Evidence set",
        full_width=True,
    )
    return review_lane_ui


@app.cell
def _(lane_lookup, review_lane_ui):
    selected_lane = "main_review"
    if review_lane_ui is not None and review_lane_ui.value is not None:
        selected_lane = str(lane_lookup.get(str(review_lane_ui.value), selected_lane))
    return selected_lane


@app.cell
def _(deliverables, section_label_lookup, selected_lane, visual_deliverables):
    visual_rows = visual_deliverables(deliverables, selected_lane=selected_lane)
    section_lookup = section_label_lookup(visual_rows)
    section_options = list(section_lookup)
    return section_lookup, section_options, visual_rows


@app.cell
def _(mo, section_options):
    deliverable_section_ui = mo.ui.dropdown(
        section_options,
        value=section_options[0] if section_options else None,
        label="Analysis section",
        full_width=True,
    )
    return deliverable_section_ui


@app.cell
def _(deliverable_section_ui, section_lookup):
    selected_section = ""
    if deliverable_section_ui is not None and deliverable_section_ui.value is not None:
        selected_section = str(section_lookup.get(str(deliverable_section_ui.value), ""))
    return selected_section


@app.cell
def _(deliverable_lookup, format_deliverable_label, section_deliverables, selected_section, visual_rows):
    section_rows = section_deliverables(visual_rows, selected_section)
    deliverable_options = [format_deliverable_label(row) for row in section_rows]
    deliverable_map = deliverable_lookup(section_rows)
    return deliverable_map, deliverable_options


@app.cell
def _(deliverable_options, mo):
    deliverable_id_ui = mo.ui.dropdown(
        deliverable_options,
        value=deliverable_options[0] if deliverable_options else None,
        label="Figure or structure view",
        full_width=True,
    )
    return deliverable_id_ui


@app.cell
def _(deliverable_id_ui, deliverable_map, deliverable_options, selected_deliverable):
    selected_label = str(deliverable_id_ui.value) if deliverable_id_ui is not None and deliverable_id_ui.value else ""
    selected_visual = selected_deliverable(
        selected_label=selected_label,
        lookup=deliverable_map,
        options=deliverable_options,
    )
    return selected_visual


@app.cell
def _(deliverables, load_structure_browser_rows, manifest_root):
    structure_browser_rows = load_structure_browser_rows(
        manifest_root=manifest_root,
        deliverables=deliverables,
    )
    return structure_browser_rows


@app.cell
def _(
    is_interactive_structure_deliverable,
    mo,
    selected_section,
    selected_visual,
    structure_browser_rows,
    structure_group_lookup,
):
    selected_visual_id = str(selected_visual.get("deliverable_id") or "") if selected_visual else ""
    if not is_interactive_structure_deliverable(selected_visual):
        structure_group_map = {}
        structure_group_ui = None
    else:
        structure_group_map = structure_group_lookup(
            structure_browser_rows,
            selected_section=selected_section,
            selected_deliverable_id=selected_visual_id,
        )
        structure_group_options = list(structure_group_map)
        structure_group_ui = mo.ui.dropdown(
            structure_group_options,
            value=structure_group_options[0] if structure_group_options else None,
            label="Structure group",
            full_width=True,
        )
    return selected_visual_id, structure_group_map, structure_group_ui


@app.cell
def _(structure_group_map, structure_group_ui):
    selected_structure_group = ""
    if structure_group_ui is not None and structure_group_ui.value:
        selected_structure_group = str(structure_group_map.get(str(structure_group_ui.value), ""))
    return selected_structure_group


@app.cell
def _(
    is_interactive_structure_deliverable,
    mo,
    selected_section,
    selected_structure_group,
    selected_visual,
    selected_visual_id,
    structure_browser_lookup,
    structure_browser_rows,
):
    if not is_interactive_structure_deliverable(selected_visual):
        structure_map = {}
        structure_ui = None
    else:
        structure_map = structure_browser_lookup(
            structure_browser_rows,
            selected_section=selected_section,
            selected_deliverable_id=selected_visual_id,
            selected_group=selected_structure_group,
        )
        structure_options = list(structure_map)
        structure_label = "Structure view"
        if structure_map:
            structure_label = str(next(iter(structure_map.values())).get("_control_label") or structure_label)
        structure_ui = mo.ui.dropdown(
            structure_options,
            value=structure_options[0] if structure_options else None,
            label=structure_label,
            full_width=True,
        )
    return structure_map, structure_ui


@app.cell
def _(structure_map, structure_ui):
    selected_structure_row = None
    if structure_ui is not None and structure_ui.value:
        selected_structure_row = structure_map.get(str(structure_ui.value))
    return selected_structure_row


@app.cell
def _(mo, selected_structure_row, structure_browser_rows, structure_highlight_lookup):
    if selected_structure_row is not None:
        structure_highlight_map = structure_highlight_lookup(
            structure_browser_rows,
            selected_row=selected_structure_row,
        )
        structure_highlight_options = list(structure_highlight_map)
        structure_highlight_ui = mo.ui.dropdown(
            structure_highlight_options,
            value=structure_highlight_options[0] if structure_highlight_options else None,
            label="Residue highlight",
            full_width=True,
        )
    else:
        structure_highlight_map = {}
        structure_highlight_ui = None
    return structure_highlight_map, structure_highlight_ui


@app.cell
def _(structure_highlight_map, structure_highlight_ui):
    selected_structure_highlight = None
    if structure_highlight_ui is not None and structure_highlight_ui.value:
        selected_structure_highlight = structure_highlight_map.get(str(structure_highlight_ui.value))
    return selected_structure_highlight


@app.cell
def _(is_interactive_structure_deliverable, mo, selected_visual):
    if not is_interactive_structure_deliverable(selected_visual):
        structure_background_ui = None
        structure_mutation_ui = None
        structure_sidechain_ui = None
        structure_protein_ui = None
        structure_dna_visible_ui = None
        structure_dna_ui = None
        structure_rna_visible_ui = None
        structure_rna_ui = None
    else:
        structure_background_ui = mo.ui.checkbox(value=True, label="Reference background")
        structure_mutation_ui = mo.ui.checkbox(value=False, label="Mutation differences")
        structure_sidechain_ui = mo.ui.checkbox(value=True, label="Side-chain sticks")
        structure_protein_ui = mo.ui.checkbox(value=False, label="Protein color")
        structure_dna_visible_ui = mo.ui.checkbox(value=True, label="Show DNA")
        structure_dna_ui = mo.ui.checkbox(value=False, label="DNA color")
        structure_rna_visible_ui = mo.ui.checkbox(value=True, label="Show RNA")
        structure_rna_ui = mo.ui.checkbox(value=False, label="RNA color")
    return (
        structure_background_ui,
        structure_dna_ui,
        structure_dna_visible_ui,
        structure_mutation_ui,
        structure_protein_ui,
        structure_rna_ui,
        structure_rna_visible_ui,
        structure_sidechain_ui,
    )


@app.cell
def _(
    structure_background_ui,
    structure_dna_ui,
    structure_dna_visible_ui,
    structure_mutation_ui,
    structure_protein_ui,
    structure_rna_ui,
    structure_rna_visible_ui,
    structure_sidechain_ui,
):
    show_reference_background = True
    show_mutation_differences = False
    show_sidechains = True
    show_dna = True
    show_rna = True
    highlight_protein = False
    highlight_dna = False
    highlight_rna = False
    if structure_background_ui is not None:
        show_reference_background = bool(structure_background_ui.value)
    if structure_mutation_ui is not None:
        show_mutation_differences = bool(structure_mutation_ui.value)
    if structure_sidechain_ui is not None:
        show_sidechains = bool(structure_sidechain_ui.value)
    if structure_dna_visible_ui is not None:
        show_dna = bool(structure_dna_visible_ui.value)
    if structure_rna_visible_ui is not None:
        show_rna = bool(structure_rna_visible_ui.value)
    if structure_protein_ui is not None:
        highlight_protein = bool(structure_protein_ui.value)
    if structure_dna_ui is not None:
        highlight_dna = bool(structure_dna_ui.value)
    if structure_rna_ui is not None:
        highlight_rna = bool(structure_rna_ui.value)
    return (
        highlight_dna,
        highlight_protein,
        highlight_rna,
        show_dna,
        show_mutation_differences,
        show_reference_background,
        show_rna,
        show_sidechains,
    )


@app.cell
def _(load_sae_feature_heatmap_manifest, manifest_root, selected_visual):
    sae_heatmap_manifest = load_sae_feature_heatmap_manifest(
        manifest_root=manifest_root,
        selected_visual=selected_visual,
    )
    return sae_heatmap_manifest


@app.cell
def _(
    is_sae_feature_heatmap_deliverable,
    mo,
    sae_heatmap_feature_lookup,
    sae_heatmap_manifest,
    selected_visual,
):
    if not is_sae_feature_heatmap_deliverable(selected_visual):
        sae_heatmap_feature_map = {}
        sae_heatmap_feature_ui = None
    else:
        sae_heatmap_feature_map = sae_heatmap_feature_lookup(sae_heatmap_manifest)
        sae_heatmap_feature_options = list(sae_heatmap_feature_map)
        sae_heatmap_feature_ui = mo.ui.dropdown(
            sae_heatmap_feature_options,
            value=sae_heatmap_feature_options[0] if sae_heatmap_feature_options else None,
            label="SAE feature",
            full_width=True,
        )
    return sae_heatmap_feature_map, sae_heatmap_feature_ui


@app.cell
def _(sae_heatmap_feature_map, sae_heatmap_feature_ui):
    selected_sae_heatmap_feature = None
    if sae_heatmap_feature_ui is not None and sae_heatmap_feature_ui.value:
        selected_sae_heatmap_feature = sae_heatmap_feature_map.get(str(sae_heatmap_feature_ui.value))
    return selected_sae_heatmap_feature


@app.cell
def _(
    deliverable_id_ui,
    deliverable_section_ui,
    is_interactive_structure_deliverable,
    is_sae_feature_heatmap_deliverable,
    manifest_root,
    mo,
    render_deliverable_details,
    render_deliverable_panel,
    render_sae_feature_heatmap,
    render_structure_browser,
    sae_heatmap_feature_ui,
    sae_heatmap_manifest,
    selected_sae_heatmap_feature,
    selected_structure_highlight,
    highlight_dna,
    highlight_protein,
    highlight_rna,
    show_dna,
    show_mutation_differences,
    show_reference_background,
    show_rna,
    show_sidechains,
    structure_background_ui,
    structure_dna_ui,
    structure_dna_visible_ui,
    structure_highlight_ui,
    selected_structure_row,
    selected_visual,
    structure_group_ui,
    structure_mutation_ui,
    structure_protein_ui,
    structure_rna_ui,
    structure_rna_visible_ui,
    structure_sidechain_ui,
    structure_ui,
    review_lane_ui,
):
    if selected_visual is None:
        panel = mo.md("No deliverable is available for the selected section.")
    else:
        if is_interactive_structure_deliverable(selected_visual):
            rendered = [
                render_structure_browser(
                    mo=mo,
                    selected_row=selected_structure_row,
                    structure_ui=structure_ui,
                    structure_group_ui=structure_group_ui,
                    structure_highlight_ui=structure_highlight_ui,
                    selected_highlight_row=selected_structure_highlight,
                    structure_background_ui=structure_background_ui,
                    structure_mutation_ui=structure_mutation_ui,
                    structure_sidechain_ui=structure_sidechain_ui,
                    structure_protein_ui=structure_protein_ui,
                    structure_dna_ui=structure_dna_ui,
                    structure_rna_ui=structure_rna_ui,
                    structure_dna_visible_ui=structure_dna_visible_ui,
                    structure_rna_visible_ui=structure_rna_visible_ui,
                    show_reference_background=show_reference_background,
                    show_mutation_differences=show_mutation_differences,
                    show_sidechains=show_sidechains,
                    show_dna=show_dna,
                    show_rna=show_rna,
                    highlight_protein=highlight_protein,
                    highlight_dna=highlight_dna,
                    highlight_rna=highlight_rna,
                ),
                render_deliverable_details(selected_visual, mo=mo),
            ]
        elif is_sae_feature_heatmap_deliverable(selected_visual):
            rendered = [
                render_sae_feature_heatmap(
                    mo=mo,
                    heatmap_manifest=sae_heatmap_manifest,
                    selected_feature_index=selected_sae_heatmap_feature,
                    feature_ui=sae_heatmap_feature_ui,
                ),
                render_deliverable_details(selected_visual, mo=mo),
            ]
        else:
            rendered = [render_deliverable_panel(selected_visual, mo=mo, manifest_root=manifest_root)]
        panel = mo.vstack(
            [
                mo.hstack(
                    [review_lane_ui, deliverable_section_ui, deliverable_id_ui],
                    justify="start",
                    align="stretch",
                    wrap=True,
                    gap=1.0,
                    widths="equal",
                ),
                *rendered,
            ],
            gap=0.45,
        )
    panel


if __name__ == "__main__":
    app.run()
"""
