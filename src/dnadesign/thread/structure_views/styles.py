"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/styles.py

Shared molecular rendering constants for structure-view backends.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.thread.structure_views.models import MoleculeClass

PROTEIN_SURFACE_COLOR = "#E8E4DA"
PROTEIN_SURFACE_OPACITY = 0.65
DNA_COLOR = "#B97700"
RNA_COLOR = "#C84C5A"

NUCLEIC_ACID_RIBBON_WIDTH = 1.35
NUCLEIC_ACID_RIBBON_THICKNESS = 0.28
NUCLEIC_ACID_SPOKE_RADIUS = 0.12

MOLECULE_CLASS_COLORS: dict[MoleculeClass, str] = {
    "protein": "#005AB5",
    "dna": DNA_COLOR,
    "rna": RNA_COLOR,
}
