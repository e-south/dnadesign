"""Study-wide molecular-scene contract for Eco1 review deliverables."""

from __future__ import annotations

from typing import Any

from dnadesign.thread.structure_views.styles import (
    DNA_COLOR,
    NUCLEIC_ACID_RIBBON_THICKNESS,
    NUCLEIC_ACID_RIBBON_WIDTH,
    PROTEIN_SURFACE_COLOR,
    PROTEIN_SURFACE_OPACITY,
    RNA_COLOR,
)

REFERENCE_MODEL_ID = "ec86kit_7v9u_reference"
CHIMERAX_SURFACE_TRANSPARENCY_PERCENT = 35
CHIMERAX_NUCLEIC_ACID_STYLE_COMMANDS = (
    "cartoon #1/D,E,F suppressBackboneDisplay true",
    "cartoon style nucleic xsect oval width 1.35 thick 0.28",
    "cartoon tether nucleic shape cylinder sides 8 scale 0.65 opacity 1",
    "show #1/D,E,F atoms",
    "nucleotides #1/D,E,F ladder",
)


def molecular_visual_contract() -> dict[str, Any]:
    """Return the renderer-neutral molecule-role contract."""

    return {
        "protein_surface_scope": "protein_only",
        "protein_surface_alpha": PROTEIN_SURFACE_OPACITY,
        "dna_color": DNA_COLOR,
        "rna_color": RNA_COLOR,
        "py3dmol_nucleic_display": "backbone_ribbon_with_base_spokes",
        "py3dmol_nucleic_ribbon_width_angstrom": NUCLEIC_ACID_RIBBON_WIDTH,
        "py3dmol_nucleic_ribbon_thickness_angstrom": NUCLEIC_ACID_RIBBON_THICKNESS,
        "chimerax_nucleic_display": "ladder",
        "chimerax_surface_transparency_percent": CHIMERAX_SURFACE_TRANSPARENCY_PERCENT,
        "chimerax_nucleotide_color_target": "acf",
    }


def reference_complex_molecule_styles(
    *,
    model_id: str = REFERENCE_MODEL_ID,
    include_protein_surface: bool,
) -> list[dict[str, Any]]:
    """Return explicit protein, DNA, and RNA styles for the retained complex."""

    styles: list[dict[str, Any]] = []
    if include_protein_surface:
        styles.append(
            {
                "model_id": model_id,
                "molecule_class": "protein",
                "label": "RT molecular surface",
                "style": "surface",
                "color": PROTEIN_SURFACE_COLOR,
                "opacity": PROTEIN_SURFACE_OPACITY,
            }
        )
    styles.extend(
        [
            {
                "model_id": model_id,
                "molecule_class": "dna",
                "label": "DNA",
                "style": "backbone_ribbon_with_base_spokes",
                "color": DNA_COLOR,
                "opacity": 1.0,
                "width": NUCLEIC_ACID_RIBBON_WIDTH,
                "thickness": NUCLEIC_ACID_RIBBON_THICKNESS,
            },
            {
                "model_id": model_id,
                "molecule_class": "rna",
                "label": "RNA",
                "style": "backbone_ribbon_with_base_spokes",
                "color": RNA_COLOR,
                "opacity": 1.0,
                "width": NUCLEIC_ACID_RIBBON_WIDTH,
                "thickness": NUCLEIC_ACID_RIBBON_THICKNESS,
            },
        ]
    )
    return styles


def chimerax_reference_complex_style_commands(*, include_protein_surface: bool = True) -> tuple[str, ...]:
    """Return the shared retained-complex ChimeraX representation commands."""

    commands = (
        "label delete",
        "hide #1 pseudobonds",
        "rename #1 eco1_rt_dna_rna_complex",
        "name protein_role #1/A",
        "name dna_role #1/D",
        "name rna_role #1/E,F",
        "hide #1 atoms",
        "cartoon #1/A",
        *CHIMERAX_NUCLEIC_ACID_STYLE_COMMANDS,
        f"color #1/A {PROTEIN_SURFACE_COLOR} target c",
        f"color #1/D {DNA_COLOR} target acf",
        f"color #1/E,F {RNA_COLOR} target acf",
    )
    if not include_protein_surface:
        return commands
    return (
        *commands,
        "surface #1/A",
        "rename #1.1 protein_surface",
        f"color #1/A {PROTEIN_SURFACE_COLOR} target s",
        f"transparency #1/A {CHIMERAX_SURFACE_TRANSPARENCY_PERCENT} target s",
    )
