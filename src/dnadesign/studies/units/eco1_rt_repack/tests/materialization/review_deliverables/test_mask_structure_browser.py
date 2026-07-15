"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_mask_structure_browser.py

Eco1 mask structure-browser tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)


def test_structure_browser_runtime_renders_mask_selection_html(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    group_lookup = structure_browser.structure_group_lookup(
        rows,
        selected_section=SECTION_CONSTRAINT_EVIDENCE,
        selected_deliverable_id="mask_structure_browser_manifest",
    )
    assert group_lookup["Fixed positions"] == "Fixed positions"
    assert group_lookup["Design spaces"] == "Design spaces"
    assert "Reference mask evidence" not in group_lookup
    assert "Design-class fixed masks" not in group_lookup
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section=SECTION_CONSTRAINT_EVIDENCE,
        selected_deliverable_id="mask_structure_browser_manifest",
        selected_group="Fixed positions",
    )
    stale_labels = (
        "Catalytic motif anchors",
        "Baseline fixed residues (clade 9 p25 + 5 A)",
        "ProteinMPNN-designable residues",
    )
    combined_labels = "\n".join(
        structure_browser.structure_browser_lookup(
            rows,
            selected_section=SECTION_CONSTRAINT_EVIDENCE,
            selected_deliverable_id="mask_structure_browser_manifest",
        )
    )
    for stale_label in stale_labels:
        assert stale_label not in combined_labels
    assert "Combined protected set | 4 residues" in lookup
    assert "NAxxH, YADD, and VTG context windows | 1 residues" in lookup
    assert "Direct DNA/RNA contacts <=5 A (Wang et al.; 7V9U) | 1 residues" in lookup
    assert "Wang thumb-contact track | 1 residues" in lookup
    selected = lookup["Combined protected set | 4 residues"]
    protected_color = str(selected["selection_styles"][0]["color"])

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<mask-highlight-dropdown>",
        structure_sidechain_ui="<side-chain-toggle>",
        structure_surface_ui="<surface-toggle>",
        structure_dna_visible_ui="<show-dna-toggle>",
        structure_rna_visible_ui="<show-rna-toggle>",
        show_protein_surface=True,
    )
    rendered_text = str(rendered)
    unescaped_rendered = html_lib.unescape(rendered_text).replace(" ", "")

    assert "<iframe" in rendered_text
    assert "3Dmol" in rendered_text
    assert "Combined protected set" in rendered_text
    assert "Fixed positions" in rendered_text
    assert "Reference selection:" not in rendered_text
    assert "No candidate structure is shown" in rendered_text
    assert "Side-chain display:" not in rendered_text
    assert "Side chains are shown only for the highlighted reference residues" in rendered_text
    assert "<side-chain-toggle>" in rendered_text
    assert "<surface-toggle>" in rendered_text
    assert "<show-dna-toggle>" in rendered_text
    assert "<show-rna-toggle>" in rendered_text
    assert "<dna-color-toggle>" not in rendered_text
    assert "<rna-color-toggle>" not in rendered_text
    assert "What this structure view shows" not in rendered_text
    assert "The protected union is fixed before ProteinMPNN samples complete sequences." in rendered_text
    assert "Interpretation limit:" not in rendered_text
    assert "does not evaluate candidate fold quality or RT activity" in rendered_text
    assert "eco1-rt-repack:reference-complex:camera-v5" in rendered_text
    assert "localStorage" in rendered_text
    assert (
        "data-selection-id=&quot;protected_union&quot;" in rendered_text
        or 'data-selection-id="protected_union"' in rendered_text
    )
    assert f'"stick":{{"color":"{protected_color}","radius":0.22}}' in unescaped_rendered
    assert 'addSurface("VDW",{"color":"#E8E4DA","opacity":0.65}' in unescaped_rendered
    assert unescaped_rendered.count("addCustom(") >= 2
    assert "addCurve(" not in unescaped_rendered
    assert '"color":"#B97700","opacity":1.0' in unescaped_rendered
    assert '"color":"#C84C5A","opacity":1.0' in unescaped_rendered
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#B97700"' in unescaped_rendered
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#C84C5A"' in unescaped_rendered
    assert '"representation":"backbone_ribbon_with_base_spokes"' in unescaped_rendered
    assert '"ribbon_width_angstrom":1.35' in unescaped_rendered
    assert '"ribbon_thickness_angstrom":0.28' in unescaped_rendered
    selection_colors = {
        str(style["color"])
        for row in rows
        if str(row.get("structure_view_mode") or "") == "reference_selection"
        and str(row.get("candidate_id") or "") == "protected_union"
        for style in row.get("selection_styles", [])
    }
    assert selection_colors == {protected_color}
