"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_structure_browser_runtime.py

Eco1 candidate structure-browser runtime tests.

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
        selected_section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        selected_deliverable_id="interactive_structure_browser_manifest",
    )
    assert "0 WT ColabFold baseline" in group_lookup
    assert "1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)" in group_lookup
    assert "2 Intermediate fold review band" in group_lookup
    lookup = structure_browser.structure_browser_lookup(
        rows,
        selected_section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        selected_deliverable_id="interactive_structure_browser_manifest",
        selected_group=group_lookup["1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)"],
    )
    selected = lookup["ProteinMPNN variant rank 1 | WT RMSD 0.82 A | pLDDT 92.4"]
    highlight_lookup = structure_browser.structure_highlight_lookup(rows, selected_row=selected)
    selected_highlight = next(row for label, row in highlight_lookup.items() if "SAE F101" in label)

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
        structure_group_ui="<structure-group-dropdown>",
        structure_highlight_ui="<sae-highlight-dropdown>",
        selected_highlight_row=selected_highlight,
        structure_sidechain_ui="<side-chain-toggle>",
        structure_protein_ui="<protein-color-toggle>",
        structure_dna_ui="<dna-color-toggle>",
        structure_rna_ui="<rna-color-toggle>",
        show_sidechains=True,
    )
    rendered_text = str(rendered)
    unescaped_rendered = html_lib.unescape(rendered_text).replace(" ", "")

    assert "<iframe" in rendered_text
    assert "3Dmol" in rendered_text
    assert "Ec86/7V9U all-atom reference" in rendered_text
    assert "ProteinMPNN variant rank 1" in rendered_text
    assert "Variant dashboard" in rendered_text
    assert "ESMC additive LLR total" in rendered_text
    assert "<sae-highlight-dropdown>" in rendered_text
    assert "<side-chain-toggle>" in rendered_text
    assert "<protein-color-toggle>" in rendered_text
    assert "<dna-color-toggle>" in rendered_text
    assert "<rna-color-toggle>" in rendered_text
    assert "Selected SAE feature" in rendered_text
    assert "F101" in rendered_text
    assert "SAE activation region" in rendered_text
    assert "Mean pLDDT" in rendered_text
    assert "Sequence identity" in rendered_text
    assert "WT-runtime CA RMSD" in rendered_text
    assert "0.82 A" in rendered_text
    assert "Browser alignment:" not in rendered_text
    assert "Side-chain display:" not in rendered_text
    assert "Candidate side-chain atoms are present and rendered as sticks" in rendered_text
    assert "The reference background includes protein side-chain atoms rendered as sticks" in rendered_text
    assert "browser_alignment_status" in rendered_text
    assert "aligned_in_memory_to_reference_ca" in rendered_text
    assert "browser_mapped_ca_rmsd" in rendered_text
    assert "reference_atom_scope" in rendered_text
    assert "sidechain_atoms_present" in rendered_text
    assert "query_atom_scope" in rendered_text
    assert "sidechain_atoms_present" in rendered_text
    assert "Raw local ColabFold PDB files are not rewritten" in rendered_text
    assert "What this structure view shows" not in rendered_text
    assert "Query coordinates are aligned in memory" in rendered_text
    assert "Interpretation limit:" not in rendered_text
    assert "ChimeraX remains the publication-still and pose-capture path" in rendered_text
    assert "eco1-rt-repack:interactive_structure_browser_manifest" in rendered_text
    assert "localStorage" in rendered_text
    assert "twoFingerPan" in rendered_text
    assert '","pdb");' in unescaped_rendered
    assert '","cif");' not in unescaped_rendered
    assert '","mmcif");' not in unescaped_rendered
    assert '"not":{"atom":["N","C","O","OXT"]}' in unescaped_rendered
    assert (
        f'"cartoon":{{"style":"rectangle","ribbon":true,"color":"{browser_colors.REFERENCE_COLOR}",'
        f'"colorfunc":function(atom){{return"{browser_colors.REFERENCE_COLOR}";}}}}' in unescaped_rendered
    )
    assert f'"color":"{browser_colors.DNA_CLASS_COLOR}"' not in unescaped_rendered
    assert f'"color":"{browser_colors.RNA_CLASS_COLOR}"' not in unescaped_rendered
    assert f'"stick":{{"color":"{browser_colors.REFERENCE_COLOR}","radius":0.16}}' in unescaped_rendered
    assert f'"stick":{{"color":"{browser_colors.CANDIDATE_PASS_COLOR}","radius":0.16}}' not in unescaped_rendered
    assert f'"stick":{{"color":"{browser_colors.RESIDUE_CATEGORY_HIGHLIGHT_COLOR}","radius":0.22}}' in (
        unescaped_rendered
    )
    assert unescaped_rendered.index(
        f'"stick":{{"color":"{browser_colors.REFERENCE_COLOR}","radius":0.16}}'
    ) < unescaped_rendered.index(
        f'"stick":{{"color":"{browser_colors.RESIDUE_CATEGORY_HIGHLIGHT_COLOR}","radius":0.22}}'
    )


def test_structure_browser_runtime_can_toggle_reference_and_mutation_overlay(tmp_path: Path) -> None:
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

    rendered = structure_browser.render_structure_browser(
        mo=FakeMo(),
        selected_row=selected,
        structure_ui="<structure-dropdown>",
        structure_group_ui="<structure-group-dropdown>",
        structure_background_ui="<reference-background-toggle>",
        structure_mutation_ui="<mutation-toggle>",
        structure_sidechain_ui="<side-chain-toggle>",
        show_reference_background=False,
        show_mutation_differences=True,
        show_sidechains=False,
    )
    rendered_text = str(rendered)
    unescaped_rendered = html_lib.unescape(rendered_text).replace(" ", "")

    assert "<reference-background-toggle>" in rendered_text
    assert "<mutation-toggle>" in rendered_text
    assert "<side-chain-toggle>" in rendered_text
    assert "Candidate differences" in rendered_text
    assert "canonical_mutations" in rendered_text
    assert "A1G, A2G" in rendered_text
    assert "Ec86/7V9U all-atom reference" not in rendered_text
    assert '"model":0,"resi":[3,4]' in unescaped_rendered
    assert f'"stick":{{"color":"{browser_colors.CANDIDATE_PASS_COLOR}","radius":0.16}}' not in unescaped_rendered
    assert "data-selection-id=&quot;candidate_differences&quot;" in rendered_text or (
        'data-selection-id="candidate_differences"' in rendered_text
    )


def test_structure_browser_manifest_rejects_missing_declared_pdb(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    full_structure_set_path = tmp_path / "foldcheck_review" / "foldcheck_full_structure_set.yaml"
    payload = yaml.safe_load(full_structure_set_path.read_text(encoding="utf-8"))
    payload["structures"][0]["local_model_artifact_path"] = "structures/full_fold_set/missing_model.pdb"
    full_structure_set_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="declared structure path is missing"):
        materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
