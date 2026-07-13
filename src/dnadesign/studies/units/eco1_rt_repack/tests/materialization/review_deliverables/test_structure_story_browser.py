"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_story_browser.py

Browser-scene contract tests for Eco1 communication visuals.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    notebook_structure_browser as structure_browser,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    load_review_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)


def test_structure_story_declares_surface_scenes_without_overwriting_evidence(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    row = next(
        item for item in manifest["deliverables"] if item["deliverable_id"] == "communication_structure_story_browser"
    )
    payload = yaml.safe_load((result.manifest_path.parent / row["path"]).read_text(encoding="utf-8"))
    scene_ids = {str(scene["candidate_id"]) for scene in payload["structures"]}
    assert {
        "native_rt_dna_rna_complex",
        "protected_catalytic_motifs",
        "protected_direct_contacts",
        "protected_conserved_positions",
        "protected_primer_recognition_context",
        "protected_union",
        "designable_peripheral_shell",
        "designable_distal_scaffold",
        "designable_combined_space",
    }.issubset(scene_ids)
    assert payload["reference"]["structure_format"] == "pdb"
    assert str(payload["reference"]["local_path"]).endswith("ec86kit_protomer1_all_atom_reference.pdb")
    assert payload["visual_contract"] == {
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
    for scene in payload["structures"]:
        assert scene["structure_format"] == "pdb"
        assert str(scene["local_path"]).endswith("ec86kit_protomer1_all_atom_reference.pdb")
        protein_styles = [
            style for style in scene.get("molecule_styles", []) if style.get("molecule_class") == "protein"
        ]
        nucleic_acid_styles = [
            style for style in scene.get("molecule_styles", []) if style.get("molecule_class") in {"dna", "rna"}
        ]
        assert any(style.get("style") == "surface" for style in protein_styles), scene["candidate_id"]
        assert all(float(style.get("opacity", 1.0)) == 0.65 for style in protein_styles), scene["candidate_id"]
        assert len(nucleic_acid_styles) == 2
        assert all(style.get("style") == "backbone_ribbon_with_base_spokes" for style in nucleic_acid_styles)
        assert all(float(style.get("width", 0.0)) == 1.35 for style in nucleic_acid_styles)
        assert all(float(style.get("thickness", 0.0)) == 0.28 for style in nucleic_acid_styles)
        assert {str(style.get("label")) for style in nucleic_acid_styles} == {"DNA", "RNA"}
        assert len({str(style.get("color")) for style in nucleic_acid_styles}) == 2
        assert str(scene.get("description") or "").strip()
    assert payload["reference"]["display_label"] == "Ec86/7V9U all-atom reference"
    assert payload["protein_surface_default"] is False


def test_structure_story_runtime_renders_surface_and_matching_nucleic_acid_colors(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        render_chimerax_png=False,
    )
    _manifest, deliverables, _manifest_path, manifest_root = load_review_manifest(str(result.notebook_path))
    deliverable_id = "communication_structure_story_browser"
    deliverable = next(row for row in deliverables if row["deliverable_id"] == deliverable_id)
    structure_rows = structure_browser.load_structure_browser_rows(
        manifest_root=manifest_root,
        deliverables=deliverables,
        selected_deliverable_id=deliverable_id,
    )
    groups = structure_browser.structure_group_lookup(
        structure_rows,
        selected_section=str(deliverable["section"]),
        selected_deliverable_id=deliverable_id,
    )
    lookup = structure_browser.structure_browser_lookup(
        structure_rows,
        selected_section=str(deliverable["section"]),
        selected_deliverable_id=deliverable_id,
        selected_group=next(iter(groups.values())),
    )
    selected_row = next(iter(lookup.values()))

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected_row,
        structure_ui="<structure>",
        structure_group_ui="<group>",
        show_sidechains=False,
    )
    rendered_text = html_lib.unescape(str(rendered)).replace(" ", "")
    assert "addSurface" not in rendered_text
    assert rendered_text.count("addCustom(") == 3
    assert "addCurve(" not in rendered_text
    assert '"color":"#B97700","opacity":1.0' in rendered_text
    assert '"color":"#C84C5A","opacity":1.0' in rendered_text
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#B97700"' in rendered_text
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#C84C5A"' in rendered_text
    assert '"representation":"backbone_ribbon_with_base_spokes"' in rendered_text
    assert '"ribbon_width_angstrom":1.35' in rendered_text
    assert '"ribbon_thickness_angstrom":0.28' in rendered_text

    without_surface = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected_row,
        structure_ui="<structure>",
        structure_group_ui="<group>",
        structure_surface_ui="<surface-toggle>",
        show_sidechains=False,
        show_protein_surface=False,
    )
    without_surface_text = html_lib.unescape(str(without_surface)).replace(" ", "")
    assert "<surface-toggle>" in str(without_surface)
    assert "addSurface" not in without_surface_text
    assert '"color":"#B97700","opacity":1.0' in without_surface_text
    assert '"color":"#C84C5A","opacity":1.0' in without_surface_text
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#B97700"' in without_surface_text
    assert '"radius":0.12,"fromCap":1,"toCap":1,"color":"#C84C5A"' in without_surface_text


@pytest.mark.parametrize(
    ("molecule_style", "message"),
    [
        (
            {
                "model_id": "reference",
                "molecule_class": "protein",
                "style": "surface",
                "color": "#E8E4DA",
                "opacity": 1.0,
            },
            "protein surfaces must use alpha 0.65",
        ),
        (
            {
                "model_id": "reference",
                "molecule_class": "dna",
                "style": "backbone_ribbon_with_base_spokes",
                "color": "#FFFFFF",
            },
            "DNA representations must use #B97700",
        ),
    ],
)
def test_structure_browser_rejects_visual_contract_drift(
    molecule_style: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        structure_browser._declared_molecule_styles(
            {"molecule_styles": [molecule_style]},
            model_ids={"reference"},
            show_protein_surface=True,
        )
