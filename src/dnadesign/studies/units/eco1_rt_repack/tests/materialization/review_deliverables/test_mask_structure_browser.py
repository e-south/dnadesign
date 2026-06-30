"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_mask_structure_browser.py

Eco1 mask structure-browser tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    mask_structure_browser,
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
    mask_row,
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
    assert group_lookup == {"Reference mask evidence": "Reference mask evidence"}
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section=SECTION_CONSTRAINT_EVIDENCE,
        selected_deliverable_id="mask_structure_browser_manifest",
        selected_group="Reference mask evidence",
    )
    selected = lookup["Protected union | 4 residues"]

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<mask-highlight-dropdown>",
        structure_sidechain_ui="<side-chain-toggle>",
        structure_dna_ui="<dna-color-toggle>",
        structure_rna_ui="<rna-color-toggle>",
    )
    rendered_text = str(rendered)

    assert "<iframe" in rendered_text
    assert "3Dmol" in rendered_text
    assert "Protected union" in rendered_text
    assert "Reference mask evidence" in rendered_text
    assert "Reference selection:" not in rendered_text
    assert "No candidate structure is shown" in rendered_text
    assert "Side-chain display:" not in rendered_text
    assert "Reference side-chain atoms are present and rendered as sticks" in rendered_text
    assert "<side-chain-toggle>" in rendered_text
    assert "<dna-color-toggle>" in rendered_text
    assert "<rna-color-toggle>" in rendered_text
    assert "What this structure view shows" not in rendered_text
    assert "All residues fixed by at least one active mask rule" in rendered_text
    assert "Interpretation limit:" not in rendered_text
    assert "does not evaluate candidate fold quality or RT activity" in rendered_text
    assert "eco1-rt-repack:mask_structure_browser_manifest" in rendered_text
    assert "localStorage" in rendered_text
    assert (
        "data-selection-id=&quot;protected&quot;" in rendered_text or 'data-selection-id="protected"' in rendered_text
    )
    selection_colors = {
        str(style["color"])
        for row in rows
        if str(row.get("structure_view_mode") or "") == "reference_selection"
        for style in row.get("selection_styles", [])
    }
    assert selection_colors == {"#D55E00"}


def test_mask_structure_browser_uses_exported_backbone_residue_numbers(tmp_path: Path) -> None:
    mask_set_path = tmp_path / "mask_set.yaml"
    mask_set_path.write_text("schema_id: thread.mask_set\nresidues: []\n", encoding="utf-8")
    reference_path = tmp_path / "proteinmpnn_request" / "chain_a_backbone.pdb"
    reference_path.parent.mkdir(parents=True)
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

    mask_structure_browser.write_mask_structure_browser_manifest(
        panel_root=tmp_path / "review_deliverables" / "structure_browser",
        mask_set_path=mask_set_path,
        reference_structure_path=reference_path,
        reference_structure_format="pdb",
        mask_residues=mask_residues,
    )

    manifest = yaml.safe_load(
        (tmp_path / "review_deliverables" / "structure_browser" / "mask_structure_browser_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    motif = next(row for row in manifest["structures"] if row["candidate_id"] == "motif_protected")
    style = motif["selection_styles"][0]
    assert style["source_coordinate_basis"] == "canonical_position"
    assert style["selection_coordinate_basis"] == "proteinmpnn_export_residue_number"
    assert style["canonical_residue_numbers"] == [3, 4]
    assert style["residue_numbers"] == [1, 2]
