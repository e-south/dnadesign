"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/structure_browser_assertions.py

Assertion helpers for Eco1 structure-browser runtime tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    structure_browser_common as browser_colors,
)


def assert_candidate_structure_browser_render(rendered_text: str, unescaped_rendered: str) -> None:
    assert "<iframe" in rendered_text
    assert "3Dmol" in rendered_text
    assert "Ec86/7V9U all-atom reference" in rendered_text
    assert "ProteinMPNN variant rank 1" in rendered_text
    assert "Variant summary" in rendered_text
    assert "ESMC additive LLR total" in rendered_text
    assert "<sae-highlight-dropdown>" in rendered_text
    assert "<side-chain-toggle>" in rendered_text
    assert "<protein-color-toggle>" in rendered_text
    assert "<show-dna-toggle>" in rendered_text
    assert "<show-rna-toggle>" in rendered_text
    assert "<dna-color-toggle>" not in rendered_text
    assert "<rna-color-toggle>" not in rendered_text
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
    assert (
        f'"cartoon":{{"style":"rectangle","ribbon":true,"color":"{browser_colors.DNA_CLASS_COLOR}",'
        f'"colorfunc":function(atom){{return"{browser_colors.DNA_CLASS_COLOR}";}}}}' in unescaped_rendered
    )
    assert (
        f'"cartoon":{{"style":"rectangle","ribbon":true,"color":"{browser_colors.RNA_CLASS_COLOR}",'
        f'"colorfunc":function(atom){{return"{browser_colors.RNA_CLASS_COLOR}";}}}}' in unescaped_rendered
    )
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


def assert_mutation_overlay_render(rendered_text: str, unescaped_rendered: str) -> None:
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


__all__ = ["assert_candidate_structure_browser_render", "assert_mutation_overlay_render"]
