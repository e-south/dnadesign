"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/__init__.py

Generic browser structure-view contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.thread.structure_views.html import render_structure_view_html, structure_view_backend_available
from dnadesign.thread.structure_views.models import (
    DNA_RESIDUE_NAMES,
    RNA_RESIDUE_NAMES,
    STANDARD_AMINO_ACID_RESIDUE_NAMES,
    StructureAtomContent,
    StructureViewModel,
    StructureViewMoleculeStyle,
    StructureViewSelectionStyle,
    StructureViewSpec,
    summarize_pdb_atom_content,
    summarize_structure_atom_content,
)

__all__ = [
    "DNA_RESIDUE_NAMES",
    "RNA_RESIDUE_NAMES",
    "STANDARD_AMINO_ACID_RESIDUE_NAMES",
    "StructureAtomContent",
    "StructureViewModel",
    "StructureViewMoleculeStyle",
    "StructureViewSelectionStyle",
    "StructureViewSpec",
    "render_structure_view_html",
    "structure_view_backend_available",
    "summarize_pdb_atom_content",
    "summarize_structure_atom_content",
]
