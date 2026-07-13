"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_mask_structure_browser_annotations.py

Eco1 RT annotation-span tests for the mask structure browser.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    mask_structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
    write_rt_annotation_context_sources,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    mask_row,
)

_RT_CONTEXT_MODULE = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rt_annotation_context"
load_rt_annotation_context = import_module(_RT_CONTEXT_MODULE).load_rt_annotation_context


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
