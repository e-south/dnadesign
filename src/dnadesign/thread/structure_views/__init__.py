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
    filter_structure_text_by_molecule_classes,
    molecule_classes_in_structure_text,
    summarize_pdb_atom_content,
    summarize_structure_atom_content,
)
from dnadesign.thread.structure_views.styles import (
    DNA_COLOR,
    MOLECULE_CLASS_COLORS,
    NUCLEIC_ACID_RIBBON_THICKNESS,
    NUCLEIC_ACID_RIBBON_WIDTH,
    NUCLEIC_ACID_SPOKE_RADIUS,
    PROTEIN_SURFACE_COLOR,
    PROTEIN_SURFACE_OPACITY,
    RNA_COLOR,
)

__all__ = [
    "DNA_RESIDUE_NAMES",
    "DNA_COLOR",
    "MOLECULE_CLASS_COLORS",
    "NUCLEIC_ACID_RIBBON_THICKNESS",
    "NUCLEIC_ACID_RIBBON_WIDTH",
    "NUCLEIC_ACID_SPOKE_RADIUS",
    "PROTEIN_SURFACE_COLOR",
    "PROTEIN_SURFACE_OPACITY",
    "RNA_RESIDUE_NAMES",
    "RNA_COLOR",
    "STANDARD_AMINO_ACID_RESIDUE_NAMES",
    "StructureAtomContent",
    "StructureViewModel",
    "StructureViewMoleculeStyle",
    "StructureViewSelectionStyle",
    "StructureViewSpec",
    "filter_structure_text_by_molecule_classes",
    "molecule_classes_in_structure_text",
    "render_structure_view_html",
    "structure_view_backend_available",
    "summarize_pdb_atom_content",
    "summarize_structure_atom_content",
]
