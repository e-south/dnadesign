"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/structure_views/test_mmcif_atom_content.py

Tests for mmCIF atom-site content summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.thread.structure_views import summarize_structure_atom_content


def test_mmcif_atom_summary_uses_declared_atom_site_column_order() -> None:
    structure_text = """\
data_permuted
loop_
_atom_site.label_comp_id
_atom_site.pdbx_PDB_ins_code
_atom_site.group_PDB
_atom_site.auth_seq_id
_atom_site.label_atom_id
_atom_site.auth_asym_id
SER ? ATOM 3 N A
SER ? ATOM 3 CA A
SER ? ATOM 3 CB A
#
"""

    content = summarize_structure_atom_content(structure_text, structure_format="mmcif")

    assert content.atom_count == 3
    assert content.residue_count == 1
    assert content.sidechain_atom_count == 1
    assert content.sidechain_residue_count == 1


def test_mmcif_atom_summary_ignores_loop_missing_required_identity_columns() -> None:
    structure_text = """\
data_missing_chain
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_seq_id
ATOM CB SER 3
#
"""

    content = summarize_structure_atom_content(structure_text, structure_format="mmcif")

    assert content.atom_count == 0
    assert content.residue_count == 0
    assert content.sidechain_atom_count == 0
    assert content.sidechain_residue_count == 0


def test_mmcif_atom_summary_parses_quoted_auth_fallback_values() -> None:
    structure_text = """\
data_quoted
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.auth_atom_id
_atom_site.label_comp_id
_atom_site.auth_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
ATOM ? "N" ? 'SER' ? 'chain A' ? "7"
ATOM ? "CA" ? 'SER' ? 'chain A' ? "7"
ATOM ? "CB" ? 'SER' ? 'chain A' ? "7"
#
"""

    content = summarize_structure_atom_content(structure_text, structure_format="mmcif")

    assert content.atom_count == 3
    assert content.residue_count == 1
    assert content.sidechain_atom_count == 1
    assert content.sidechain_residue_count == 1


def test_mmcif_atom_summary_finds_atom_site_data_across_multiple_loops() -> None:
    structure_text = """\
data_multiple_loops
loop_
_entity.id
_entity.type
1 polymer
#
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
ATOM N ALA A 2 A
ATOM CA ALA A 2 A
#
loop_
_atom_site.pdbx_PDB_ins_code
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.auth_comp_id
_atom_site.auth_atom_id
_atom_site.group_PDB
B 2 A ALA CB ATOM
#
"""

    content = summarize_structure_atom_content(structure_text, structure_format="mmcif")

    assert content.atom_count == 3
    assert content.residue_count == 2
    assert content.sidechain_atom_count == 1
    assert content.sidechain_residue_count == 1
