"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/structure_views/test_mmcif_molecule_classes.py

Tests for header-aware mmCIF molecule classification and filtering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.thread.structure_views import (
    filter_structure_text_by_molecule_classes,
    molecule_classes_in_structure_text,
)

_PERMUTED_MOLECULE_MMCIF = """\
data_permuted_molecules
loop_
_atom_site.label_comp_id
_atom_site.pdbx_PDB_ins_code
_atom_site.group_PDB
_atom_site.auth_seq_id
_atom_site.label_atom_id
_atom_site.auth_asym_id
SER ? ATOM 3 CA A
DA ? HETATM 1 P D
U ? HETATM 2 P R
#
"""


def test_mmcif_molecule_classes_use_declared_atom_site_column_order() -> None:
    assert molecule_classes_in_structure_text(
        _PERMUTED_MOLECULE_MMCIF,
        structure_format="mmcif",
    ) == frozenset({"protein", "dna", "rna"})


def test_mmcif_molecule_filter_uses_declared_atom_site_column_order() -> None:
    dna_text = filter_structure_text_by_molecule_classes(
        _PERMUTED_MOLECULE_MMCIF,
        structure_format="mmcif",
        visible_molecule_classes=("dna",),
    )

    assert "SER ? ATOM 3 CA A" not in dna_text
    assert "DA ? HETATM 1 P D" in dna_text
    assert "U ? HETATM 2 P R" not in dna_text
    assert molecule_classes_in_structure_text(
        dna_text,
        structure_format="mmcif",
    ) == frozenset({"dna"})
