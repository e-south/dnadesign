"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_mask_structure_browser_coordinates.py

Eco1 mask structure-browser coordinate tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    mask_structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
    write_rt_annotation_context_sources,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    mask_row,
)

_RT_CONTEXT_MODULE = (
    "dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rt_annotation_context"
)
load_rt_annotation_context = import_module(_RT_CONTEXT_MODULE).load_rt_annotation_context


def test_mask_structure_browser_uses_exported_backbone_residue_numbers(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    mask_set_path = tmp_path / "mask_set.yaml"
    reference_path = tmp_path / "proteinmpnn_request" / "chain_a_backbone.pdb"
    reference_path.write_text(
        "ATOM      1  CA  SER A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  GLU A   3       3.000   0.000   0.000  1.00  0.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    mask_residues = [
        mask_row(1, mapped=False),
        mask_row(2, mapped=False),
        mask_row(3, motif=True),
        mask_row(4, motif=True),
        mask_row(5, protected=True),
    ]
    annotation_tracks_path, manual_authority_path = write_rt_annotation_context_sources(tmp_path)
    rt_annotation_context = load_rt_annotation_context(
        annotation_tracks_path=annotation_tracks_path,
        manual_mask_authority_source_path=manual_authority_path,
    )

    mask_structure_browser.write_mask_structure_browser_manifest(
        panel_root=tmp_path / "review_deliverables" / "structure_browser",
        mask_set_path=mask_set_path,
        design_classes_root=tmp_path / "design_classes",
        reference_structure_path=reference_path,
        reference_structure_format="pdb",
        mask_residues=mask_residues,
        rt_annotation_context=rt_annotation_context,
    )

    manifest = yaml.safe_load(
        (tmp_path / "review_deliverables" / "structure_browser" / "mask_structure_browser_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    fixed_mask = next(
        row for row in manifest["structures"] if row["candidate_id"] == "eco1_rt_clade9_plurality25_contact5a_v1"
    )
    style = fixed_mask["selection_styles"][0]
    assert style["source_coordinate_basis"] == "canonical_position"
    assert style["selection_coordinate_basis"] == "proteinmpnn_export_residue_number"
    assert style["canonical_residue_numbers"] == [2, 3, 4, 5]
    assert style["residue_numbers"] == [1, 2, 3]
