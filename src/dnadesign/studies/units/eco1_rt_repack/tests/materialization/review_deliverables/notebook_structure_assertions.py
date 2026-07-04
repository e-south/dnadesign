"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/notebook_structure_assertions.py

Notebook structure-browser control assertions for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def assert_structure_notebook_contract(*, notebook_text: str, combined_text: str) -> None:
    """Assert the generated notebook exposes stable, plain structure controls."""

    assert "structure_sidechain_ui = mo.ui.checkbox" in notebook_text
    assert 'label="Side-chain sticks"' in notebook_text
    assert "structure_protein_ui = mo.ui.checkbox" in notebook_text
    assert 'label="Protein color"' in notebook_text
    assert "structure_dna_ui = mo.ui.checkbox" in notebook_text
    assert 'structure_dna_ui = mo.ui.checkbox(value=False, label="DNA color")' in notebook_text
    assert 'label="DNA color"' in notebook_text
    assert "structure_dna_visible_ui = mo.ui.checkbox" in notebook_text
    assert 'label="Show DNA"' in notebook_text
    assert "structure_rna_ui = mo.ui.checkbox" in notebook_text
    assert 'structure_rna_ui = mo.ui.checkbox(value=False, label="RNA color")' in notebook_text
    assert 'label="RNA color"' in notebook_text
    assert "structure_rna_visible_ui = mo.ui.checkbox" in notebook_text
    assert 'label="Show RNA"' in notebook_text
    assert "WT Ec86 control" not in combined_text
    assert "is_interactive_structure_deliverable(" in combined_text
    assert "render_deliverable_details(" in combined_text
    assert 'if selected_section == "fold_review":' not in notebook_text
    assert "Reference sequence, alignment, and mask" not in combined_text
    assert "Reference scaffold and mask evidence" not in combined_text
    assert "ColabFold structure triage" not in combined_text
    assert 'structure_label = "Structure view"' in notebook_text
    assert "label=structure_label" in notebook_text
    assert "structure_group_ui = mo.ui.dropdown" in notebook_text
    assert 'structure_group_label = "Structure group"' in notebook_text
    assert 'structure_group_label = "Mask evidence category"' in notebook_text
    assert 'structure_group_label = "Design class"' in notebook_text
    assert "label=structure_group_label" in notebook_text
    assert "if not is_interactive_structure_deliverable(selected_visual):" in notebook_text
    assert "structure_group_ui = None" in notebook_text
    assert "structure_background_ui = mo.ui.checkbox" in notebook_text
    assert "structure_background_ui = None" not in notebook_text
    assert 'label="Reference background"' in notebook_text
    assert "structure_mutation_ui = mo.ui.checkbox" in notebook_text
    assert "structure_mutation_ui = None" not in notebook_text
    assert 'label="Mutation differences"' in notebook_text
    assert "structure_dna_visible_ui = None" not in notebook_text
    assert "structure_rna_visible_ui = None" not in notebook_text
    assert "show_sidechains" in notebook_text
    assert "highlight_dna" in notebook_text
    assert "highlight_rna" in notebook_text
    assert "highlight_protein" in notebook_text
    assert "show_dna" in notebook_text
    assert "show_rna" in notebook_text
    assert "selected_deliverable_id=selected_visual_id" in notebook_text
    assert "selected_group=selected_structure_group" in notebook_text
    assert "structure_group_lookup" in combined_text
