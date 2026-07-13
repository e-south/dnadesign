"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/structure_scenes.py

Residue-set scenes shared by browser and ChimeraX structure stories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)

from .style import (
    CONSERVATION_COLOR,
    CONTACT_COLOR,
    MOTIF_COLOR,
    POLICY_COLORS,
    PROTECTED_COLOR,
    PROTEIN_SURFACE_COLOR,
    RECOGNITION_COLOR,
)


def structure_scene_specs(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Return the ordered evidence and design-space scenes for one structure."""

    context_rows = _context_rows(rows)
    protected_union = {
        position
        for position, row in context_rows.items()
        if bool(row.get("protected_reason_codes")) or not bool(row.get("is_designable_backbone_position"))
    }
    return (
        _scene(
            "native_rt_dna_rna_complex",
            "Native RT-DNA-RNA complex",
            "1 Native complex",
            "The retained complex provides the coordinate frame for every design rule.",
            set(),
            PROTEIN_SURFACE_COLOR,
        ),
        _scene(
            "protected_catalytic_motifs",
            "NAxxH, YADD, and VTG contexts (Wang et al.)",
            "2 Protected evidence",
            "The NAxxH, YADD, and VTG contexts remain fixed during sequence generation.",
            _positions_with_value(context_rows, "motif_context_codes"),
            MOTIF_COLOR,
        ),
        _scene(
            "protected_direct_contacts",
            "Direct DNA/RNA contacts <=5 A (Wang et al.; 7V9U)",
            "2 Protected evidence",
            "Residues within 5 A of retained DNA or RNA remain fixed.",
            _positions_with_value(context_rows, "is_direct_contact_le_5a"),
            CONTACT_COLOR,
        ),
        _scene(
            "protected_conserved_positions",
            "WT is clade-9 plurality at >=25%",
            "2 Protected evidence",
            "WT-plurality positions passing the declared clade-9 conservation rule remain fixed.",
            _positions_with_value(context_rows, "is_conserved_core"),
            CONSERVATION_COLOR,
        ),
        _scene(
            "protected_primer_recognition_context",
            "Primer-RNA recognition 255-311 (Inouye et al.)",
            "2 Protected evidence",
            "Mapped residues 255-311 remain fixed as the declared primer-recognition context.",
            _positions_with_value(context_rows, "is_c_terminal_thumb_context"),
            RECOGNITION_COLOR,
        ),
        _scene(
            "protected_union",
            "Combined protected set",
            "2 Protected evidence",
            "The protected union is fixed before ProteinMPNN samples complete sequences.",
            protected_union,
            PROTECTED_COLOR,
        ),
        _scene(
            "designable_peripheral_shell",
            "Open peripheral shell >5 to <=10 A",
            "3 Design spaces",
            "Eligible peripheral residues can sample MSA-observed, non-acidifying alternatives.",
            _open_positions(rows, NEAR_DNA_RNA_ACID_FREE_POLICY_ID),
            POLICY_COLORS[NEAR_DNA_RNA_ACID_FREE_POLICY_ID],
        ),
        _scene(
            "designable_distal_scaffold",
            "Open distal scaffold >10 A",
            "3 Design spaces",
            "The N-terminal-enriched distal set provides a distant-repacking comparison, not a direct contact "
            "hypothesis.",
            _open_positions(rows, DISTAL_SCAFFOLD_POLICY_ID),
            POLICY_COLORS[DISTAL_SCAFFOLD_POLICY_ID],
        ),
        _scene(
            "designable_combined_space",
            "Open combined design space",
            "3 Design spaces",
            "The combined policy samples peripheral and distal positions jointly in each complete sequence.",
            _open_positions(rows, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID),
            POLICY_COLORS[COMBINED_NEAR_PLUS_DISTAL_POLICY_ID],
        ),
    )


def _scene(
    scene_id: str,
    label: str,
    group: str,
    description: str,
    positions: set[int],
    color: str,
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "label": label,
        "group": group,
        "description": description,
        "positions": positions,
        "color": color,
    }


def _context_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    combined = [row for row in rows if str(row.get("policy_id") or "") == COMBINED_NEAR_PLUS_DISTAL_POLICY_ID]
    return {int(row["eco1_position"]): row for row in (combined or rows)}


def _positions_with_value(rows: dict[int, dict[str, Any]], field: str) -> set[int]:
    return {position for position, row in rows.items() if bool(row.get(field))}


def _open_positions(rows: list[dict[str, Any]], policy_id: str) -> set[int]:
    return {
        int(row["eco1_position"])
        for row in rows
        if str(row.get("policy_id") or "") == policy_id and bool(row.get("is_open_position"))
    }
