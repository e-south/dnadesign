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
    """Write a compact manifest-backed marimo notebook for Eco1 review deliverables."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_NOTEBOOK_SOURCE, encoding="utf-8")


_NOTEBOOK_SOURCE = """import marimo

__generated_with = "dnadesign.eco1_rt_repack.review_deliverables"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
        deliverable_lookup,
        format_deliverable_label,
        load_review_manifest,
        render_deliverable_panel,
        render_intro,
        section_deliverables,
        section_label_lookup,
        selected_deliverable,
        visual_deliverables,
    )
    from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
        notebook_sae_features as sae_runtime,
        notebook_structure_browser as structure_runtime,
    )

    load_sae_top_feature_rows = sae_runtime.load_sae_top_feature_rows
    load_structure_browser_rows = structure_runtime.load_structure_browser_rows
    render_sae_feature_inspector = sae_runtime.render_sae_feature_inspector
    render_structure_browser = structure_runtime.render_structure_browser
    sae_feature_lookup = sae_runtime.sae_feature_lookup
    sae_protein_lookup = sae_runtime.sae_protein_lookup
    selected_sae_feature_rows = sae_runtime.selected_sae_feature_rows
    structure_browser_lookup = structure_runtime.structure_browser_lookup

    return (
        deliverable_lookup,
        format_deliverable_label,
        load_review_manifest,
        load_sae_top_feature_rows,
        load_structure_browser_rows,
        mo,
        render_deliverable_panel,
        render_intro,
        render_sae_feature_inspector,
        render_structure_browser,
        sae_feature_lookup,
        sae_protein_lookup,
        section_deliverables,
        section_label_lookup,
        selected_deliverable,
        selected_sae_feature_rows,
        structure_browser_lookup,
        visual_deliverables,
    )


@app.cell
def _(load_review_manifest):
    manifest, deliverables, manifest_path, manifest_root = load_review_manifest(__file__)
    return deliverables, manifest, manifest_path, manifest_root


@app.cell
def _(mo, render_intro):
    render_intro(mo)


@app.cell
def _(deliverables, section_label_lookup, visual_deliverables):
    visual_rows = visual_deliverables(deliverables)
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
        label="Visual",
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
def _(deliverables, load_sae_top_feature_rows, manifest_root):
    sae_top_feature_rows = load_sae_top_feature_rows(
        manifest_root=manifest_root,
        deliverables=deliverables,
    )
    return sae_top_feature_rows


@app.cell
def _(deliverables, load_structure_browser_rows, manifest_root):
    structure_browser_rows = load_structure_browser_rows(
        manifest_root=manifest_root,
        deliverables=deliverables,
    )
    return structure_browser_rows


@app.cell
def _(mo, selected_section, structure_browser_lookup, structure_browser_rows):
    structure_map = structure_browser_lookup(structure_browser_rows, selected_section=selected_section)
    structure_options = list(structure_map)
    structure_ui = mo.ui.dropdown(
        structure_options,
        value=structure_options[0] if structure_options else None,
        label="Structure",
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
def _(mo, sae_protein_lookup, sae_top_feature_rows, selected_section):
    protein_map = sae_protein_lookup(sae_top_feature_rows, selected_section=selected_section)
    protein_options = list(protein_map)
    sae_protein_ui = mo.ui.dropdown(
        protein_options,
        value=protein_options[0] if protein_options else None,
        label="Protein",
        full_width=True,
    )
    return protein_map, sae_protein_ui


@app.cell
def _(protein_map, sae_protein_ui, sae_top_feature_rows, selected_sae_feature_rows):
    selected_sae_protein = ""
    if sae_protein_ui is not None and sae_protein_ui.value:
        selected_sae_protein = str(protein_map.get(str(sae_protein_ui.value), ""))
    selected_feature_rows = selected_sae_feature_rows(
        sae_top_feature_rows,
        candidate_id=selected_sae_protein,
    )
    return selected_feature_rows, selected_sae_protein


@app.cell
def _(mo, sae_feature_lookup, selected_feature_rows):
    feature_map = sae_feature_lookup(selected_feature_rows)
    feature_options = list(feature_map)
    sae_feature_ui = mo.ui.dropdown(
        feature_options,
        value=feature_options[0] if feature_options else None,
        label="SAE feature",
        full_width=True,
    )
    return feature_map, sae_feature_ui


@app.cell
def _(feature_map, sae_feature_ui):
    selected_sae_feature = None
    if sae_feature_ui is not None and sae_feature_ui.value:
        selected_sae_feature = feature_map.get(str(sae_feature_ui.value))
    return selected_sae_feature


@app.cell
def _(
    deliverable_id_ui,
    deliverable_section_ui,
    deliverables,
    manifest_root,
    mo,
    render_deliverable_panel,
    render_sae_feature_inspector,
    render_structure_browser,
    sae_feature_ui,
    sae_protein_ui,
    selected_section,
    selected_sae_feature,
    selected_structure_row,
    selected_visual,
    structure_ui,
):
    if selected_visual is None:
        panel = mo.md("No deliverable is available for the selected section.")
    else:
        rendered = [render_deliverable_panel(selected_visual, mo=mo, manifest_root=manifest_root)]
        if selected_section == "biohub_esmc_sae_interpretation":
            rendered.append(
                render_sae_feature_inspector(
                    mo=mo,
                    manifest_root=manifest_root,
                    deliverables=deliverables,
                    selected_row=selected_sae_feature,
                    protein_ui=sae_protein_ui,
                    feature_ui=sae_feature_ui,
                )
            )
        if selected_section == "fold_review":
            rendered.append(
                render_structure_browser(
                    mo=mo,
                    selected_row=selected_structure_row,
                    structure_ui=structure_ui,
                )
            )
        panel = mo.vstack(
            [mo.hstack([deliverable_section_ui, deliverable_id_ui], justify="start", gap=1.0), *rendered],
            gap=0.45,
        )
    panel


if __name__ == "__main__":
    app.run()
"""
