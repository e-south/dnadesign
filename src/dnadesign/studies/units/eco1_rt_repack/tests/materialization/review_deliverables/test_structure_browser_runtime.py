"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_browser_runtime.py

Eco1 interactive structure-browser runtime tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.structure_browser import (
    write_mask_structure_browser_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
    mask_row,
)


def test_structure_browser_runtime_renders_py3dmol_html(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    group_lookup = structure_browser.structure_group_lookup(
        rows,
        selected_section="design_and_fold_triage",
        selected_deliverable_id="interactive_structure_browser_manifest",
    )
    assert "WT baseline" in group_lookup
    assert "Low-deviation fold-check candidates" in group_lookup
    assert "Other fold-check candidates" in group_lookup
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section="design_and_fold_triage",
        selected_deliverable_id="interactive_structure_browser_manifest",
        selected_group=group_lookup["Low-deviation fold-check candidates"],
    )
    selected = lookup["ProteinMPNN variant rank 1 | WT RMSD 0.82 A | pLDDT 92.4"]

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
        structure_group_ui="<structure-group-dropdown>",
    )
    rendered_text = str(rendered)

    assert "<iframe" in rendered_text
    assert "3Dmol" in rendered_text
    assert "ec86kit/7V9U reference" in rendered_text
    assert "ProteinMPNN variant rank 1" in rendered_text
    assert "Structure metric summary" in rendered_text
    assert "Mean pLDDT" in rendered_text
    assert "Sequence identity" in rendered_text
    assert "WT-runtime C-alpha RMSD 0.82 A" in rendered_text
    assert "Browser alignment:" in rendered_text
    assert "browser_alignment_status" in rendered_text
    assert "aligned_in_memory_to_reference_ca" in rendered_text
    assert "browser_mapped_ca_rmsd" in rendered_text
    assert "Raw local ColabFold PDB files are not rewritten" in rendered_text


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
        selected_section="scaffold_and_mask",
        selected_deliverable_id="mask_structure_browser_manifest",
    )
    assert group_lookup == {"Reference mask evidence": "Reference mask evidence"}
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section="scaffold_and_mask",
        selected_deliverable_id="mask_structure_browser_manifest",
        selected_group="Reference mask evidence",
    )
    selected = lookup["Protected union | 4 residues"]

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<mask-highlight-dropdown>",
    )
    rendered_text = str(rendered)

    assert "<iframe" in rendered_text
    assert "3Dmol" in rendered_text
    assert "Protected union" in rendered_text
    assert "Reference mask evidence" in rendered_text
    assert "Reference selection:" in rendered_text
    assert "No candidate structure is shown" in rendered_text
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

    write_mask_structure_browser_manifest(
        panel_root=tmp_path / "review_deliverables" / "structure_browser",
        mask_set_path=mask_set_path,
        reference_backbone_path=reference_path,
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


def test_structure_browser_manifest_rejects_missing_declared_pdb(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    full_structure_set_path = tmp_path / "foldcheck_review" / "foldcheck_full_structure_set.yaml"
    payload = yaml.safe_load(full_structure_set_path.read_text(encoding="utf-8"))
    payload["structures"][0]["local_model_artifact_path"] = "structures/full_fold_set/missing_model.pdb"
    full_structure_set_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="declared structure path is missing"):
        materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
