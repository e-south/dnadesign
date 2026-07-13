"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_browser_molecule_colors.py

Eco1 structure-browser molecule-color tests.

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
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    structure_browser_common as browser_colors,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.runtime_fixtures import (
    FakeMo,
)


def test_structure_browser_runtime_uses_fixed_molecule_class_colors(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    rows = structure_browser.load_structure_browser_rows(
        manifest_root=result.manifest_path.parent,
        deliverables=manifest["deliverables"],
    )
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        selected_deliverable_id="interactive_structure_browser_manifest",
        selected_group="1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)",
    )
    selected = lookup["ProteinMPNN variant rank 1 | WT RMSD 0.82 A | pLDDT 92.4"]

    default_rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
        structure_group_ui="<structure-group-dropdown>",
        structure_dna_visible_ui="<show-dna-toggle>",
        structure_rna_visible_ui="<show-rna-toggle>",
        show_dna=True,
        show_rna=True,
        highlight_dna=True,
        highlight_rna=True,
    )
    default_text = str(default_rendered)
    default_unescaped = html_lib.unescape(default_text).replace(" ", "")

    assert '"resn":["DA","DC","DG","DT"]' in default_unescaped
    assert '"resn":["A","C","G","I","U"]' in default_unescaped
    assert default_unescaped.count("addCustom(") == 3
    assert "addCurve(" not in default_unescaped
    assert f'"color":"{browser_colors.DNA_CLASS_COLOR}","opacity":1.0' in default_unescaped
    assert f'"color":"{browser_colors.RNA_CLASS_COLOR}","opacity":1.0' in default_unescaped
    assert f'"radius":0.12,"fromCap":1,"toCap":1,"color":"{browser_colors.DNA_CLASS_COLOR}"' in (default_unescaped)
    assert f'"radius":0.12,"fromCap":1,"toCap":1,"color":"{browser_colors.RNA_CLASS_COLOR}"' in (default_unescaped)
    assert "<protein-color-toggle>" not in default_text
    assert f'"cartoon":{{"color":"{browser_colors.REFERENCE_COLOR}"}}' in default_unescaped
    assert f'"stick":{{"color":"{browser_colors.PROTEIN_CLASS_COLOR}","radius":0.16}}' not in default_unescaped

    hidden_dna_rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
        structure_group_ui="<structure-group-dropdown>",
        structure_dna_visible_ui="<show-dna-toggle>",
        structure_rna_visible_ui="<show-rna-toggle>",
        show_dna=False,
        show_rna=True,
        highlight_dna=True,
        highlight_rna=True,
    )
    hidden_text = str(hidden_dna_rendered)
    hidden_unescaped = html_lib.unescape(hidden_text).replace(" ", "")

    assert "<show-dna-toggle>" in hidden_text
    assert "<show-rna-toggle>" in hidden_text
    assert '"resn":["DA","DC","DG","DT"]' not in hidden_unescaped
    assert f'"color":"{browser_colors.DNA_CLASS_COLOR}","opacity":1.0' not in hidden_unescaped
    assert " DA B" not in hidden_text
    assert '"resn":["A","C","G","I","U"]' in hidden_unescaped
    assert hidden_unescaped.count("addCustom(") == 2
    assert "addCurve(" not in hidden_unescaped
    assert f'"color":"{browser_colors.RNA_CLASS_COLOR}","opacity":1.0' in hidden_unescaped
    assert "Molecule visibility" in hidden_text
    assert "Protein, RNA." in hidden_text


def test_browser_reference_regenerates_stale_pdb_without_dropping_nucleic_sugars(tmp_path: Path) -> None:
    structure_root = tmp_path / "foldcheck_review" / "structures"
    structure_root.mkdir(parents=True)
    reference_backbone_path = structure_root / "ec86kit_chain_a_backbone_reference.pdb"
    reference_backbone_path.write_text("END\n", encoding="utf-8")
    mmcif_path = structure_root / "ec86kit_protomer1_all_atom_reference.cif"
    mmcif_path.write_text(
        "\n".join(
            (
                "data_reference",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_PDB_model_num",
                "ATOM 1 P P . DG C 2 20 96.895 154.240 135.675 D 1 ? 1.00 56.79 1",
                "ATOM 2 O O5' . DG C 2 20 98.131 153.280 135.351 D 1 ? 1.00 56.79 1",
                "ATOM 3 C C1' . DG C 2 20 101.222 151.135 135.285 D 1 ? 1.00 56.79 1",
                "ATOM 4 N N9 . DG C 2 20 101.078 151.472 136.690 D 1 ? 1.00 56.79 1",
                "#",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    stale_pdb_path = structure_root / "ec86kit_protomer1_all_atom_reference.pdb"
    stale_pdb_path.write_text(
        "ATOM      1  P    DG D   1       0.000   0.000   0.000  1.00 10.00           P\nEND\n",
        encoding="utf-8",
    )

    staged = browser_colors.stage_browser_reference_structure(
        repo_root=tmp_path,
        reference_backbone_path=reference_backbone_path,
    )

    pdb_text = staged.local_path.read_text(encoding="utf-8")
    assert staged.source_status == "regenerated_browser_pdb_from_all_atom_mmcif"
    assert sum(line.startswith(("ATOM  ", "HETATM")) for line in pdb_text.splitlines()) == 4
    assert " O5' " in pdb_text
    assert " C1' " in pdb_text
    assert " N9  " in pdb_text
