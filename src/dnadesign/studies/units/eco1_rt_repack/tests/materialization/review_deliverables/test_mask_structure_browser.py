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
from importlib import import_module
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    mask_structure_browser,
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    structure_browser_common as browser_colors,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
    write_rt_annotation_context_sources,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
    mask_row,
)

_RT_CONTEXT_MODULE = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rt_annotation_context"
load_rt_annotation_context = import_module(_RT_CONTEXT_MODULE).load_rt_annotation_context


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
    assert group_lookup["Current mask evidence"] == "Current mask evidence"
    assert "Reference mask evidence" not in group_lookup
    assert "Design-class fixed masks" not in group_lookup
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section=SECTION_CONSTRAINT_EVIDENCE,
        selected_deliverable_id="mask_structure_browser_manifest",
        selected_group="Current mask evidence",
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
    assert "Protected residues | 4 residues" in lookup
    assert "Catalytic and retron motif anchors | 1 residues" in lookup
    assert "Wang/Ec86 substrate-contact priors | 1 residues" in lookup
    selected = lookup["Protected residues | 4 residues"]

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
    assert "Protected residues" in rendered_text
    assert "Current mask evidence" in rendered_text
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
    assert "Residues fixed by the current Eco1 RT mask rule." in rendered_text
    assert "Interpretation limit:" not in rendered_text
    assert "does not evaluate candidate fold quality or RT activity" in rendered_text
    assert "eco1-rt-repack:reference-complex:camera-v5" in rendered_text
    assert "localStorage" in rendered_text
    assert (
        "data-selection-id=&quot;active_mask_protected_positions&quot;" in rendered_text
        or 'data-selection-id="active_mask_protected_positions"' in rendered_text
    )
    assert f'"stick":{{"color":"{browser_colors.RESIDUE_CATEGORY_HIGHLIGHT_COLOR}","radius":0.22}}' in (
        unescaped_rendered
    )
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
        and str(row.get("candidate_id") or "") == "active_mask_protected_positions"
        for style in row.get("selection_styles", [])
    }
    assert selection_colors == {browser_colors.RESIDUE_CATEGORY_HIGHLIGHT_COLOR}


def test_mask_structure_browser_exposes_rt_annotation_spans_as_reference_highlights(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    annotation_tracks_path, manual_authority_path = write_rt_annotation_context_sources(tmp_path)
    rt_annotation_context = load_rt_annotation_context(
        annotation_tracks_path=annotation_tracks_path,
        manual_mask_authority_source_path=manual_authority_path,
    )

    deliverable = mask_structure_browser.write_mask_structure_browser_manifest(
        panel_root=tmp_path / "review_deliverables" / "structure_browser",
        mask_set_path=tmp_path / "mask_set.yaml",
        reference_structure_path=tmp_path / "proteinmpnn_request" / "chain_a_backbone.pdb",
        reference_structure_format="pdb",
        mask_residues=[mask_row(position, protected=True) for position in range(1, 7)],
        rt_annotation_context=rt_annotation_context,
        policy_position_rows=pq.read_table(
            tmp_path / "generation_policies_v3" / "generation_policy_positions.parquet"
        ).to_pylist(),
        policy_positions_path=tmp_path / "generation_policies_v3" / "generation_policy_positions.parquet",
    )

    manifest = yaml.safe_load(
        (tmp_path / "review_deliverables" / "structure_browser" / "mask_structure_browser_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["visual_contract"] == {
        "protein_surface_scope": "protein_only",
        "protein_surface_alpha": 0.65,
        "dna_color": "#B97700",
        "rna_color": "#C84C5A",
        "py3dmol_nucleic_display": "backbone_ribbon_with_base_spokes",
        "py3dmol_nucleic_ribbon_width_angstrom": 1.35,
        "py3dmol_nucleic_ribbon_thickness_angstrom": 0.28,
        "chimerax_nucleic_display": "ladder",
        "chimerax_surface_transparency_percent": 35,
        "chimerax_nucleotide_color_target": "acf",
    }
    assert manifest["protein_surface_default"] is False
    rows_by_id = {row["candidate_id"]: row for row in manifest["structures"]}
    rt1 = rows_by_id["rt1_interval"]
    region_x = rows_by_id["retron_x_context"]

    assert rt1["display_label"] == "RT1"
    assert rt1["group"] == "RT annotation spans"
    assert rt1["structure_view_mode"] == "reference_selection"
    assert rt1["selection_residue_count"] == 2
    assert rt1["selection_styles"][0]["canonical_residue_numbers"] == [2, 3]
    assert rt1["selection_styles"][0]["residue_numbers"] == [2, 3]
    assert rt1["selection_styles"][0]["selection_id"] == "rt1_interval"
    assert "display-only rt annotation" in rt1["description"].lower()
    assert region_x["selection_styles"][0]["canonical_residue_numbers"] == [2, 3, 4]
    assert "rt_annotation_tracks" in manifest["source_hashes"]
    for row in manifest["structures"]:
        styles = {style["molecule_class"]: style for style in row["molecule_styles"]}
        assert styles["protein"]["style"] == "surface"
        assert styles["protein"]["opacity"] == 0.65
        assert styles["dna"]["color"] == "#B97700"
        assert styles["rna"]["color"] == "#C84C5A"

    rows = structure_browser.load_structure_browser_rows(
        manifest_root=tmp_path,
        deliverables=[deliverable],
    )
    highlight_lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section=SECTION_CONSTRAINT_EVIDENCE,
        selected_deliverable_id="mask_structure_browser_manifest",
        selected_group="RT annotation spans",
    )
    assert "RT1 | 2 residues" in highlight_lookup
    assert "RT2 | 2 residues" in highlight_lookup
    assert "Region X local context | 3 residues" in highlight_lookup
    assert "Catalytic YADD local context | 3 residues" in highlight_lookup
    assert "NAxxH | 1 residues" in highlight_lookup
    assert "YADD | 1 residues" in highlight_lookup
