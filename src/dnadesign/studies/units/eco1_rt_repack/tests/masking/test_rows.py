"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/masking/test_rows.py

Mask-row algebra tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.masking import (
    compose_mask_rows,
    summarize_mask_rows,
)


def test_compose_mask_rows_uses_clade9_plurality25_direct_contact5a_policy() -> None:
    rows = compose_mask_rows(
        residue_rows=[
            _residue(1, "mapped"),
            _residue(2, "mapped"),
            _residue(3, "missing_structure"),
            _residue(4, "mapped"),
            _residue(5, "mapped"),
            _residue(6, "mapped"),
            _residue(7, "mapped"),
        ],
        contact_geometry_rows=[
            _contact_geometry(1, distance=21.0),
            _contact_geometry(2, distance=21.0),
            _contact_geometry(3, distance=None),
            _contact_geometry(4, distance=5.0),
            _contact_geometry(5, distance=5.1),
            _contact_geometry(6, distance=21.0),
            _contact_geometry(7, distance=21.0),
        ],
        conservation_rows=[
            _conservation(1, profile_id="ec86_clade9_conservation_v1", passes=False),
            _conservation(2, profile_id="ec86_clade9_conservation_v1", passes=False),
            _conservation(3, profile_id="ec86_clade9_conservation_v1", passes=False),
            _conservation(4, profile_id="ec86_clade9_conservation_v1", passes=False),
            _conservation(5, profile_id="ec86_clade9_conservation_v1", passes=False),
            _conservation(6, profile_id="ec86_iia3_cluster42_1_conservation_v1", passes=True),
            _conservation(7, profile_id="ec86_clade9_conservation_v1", passes=True),
        ],
        manual_authority={
            "residues": [
                {
                    "canonical_position": 1,
                    "manual_mask_reason": "test_manual_anchor",
                }
            ],
            "candidate_prior_residues": [
                {
                    "canonical_position": 2,
                    "reason": "test_wang_direct_contact",
                }
            ],
        },
    )

    by_position = {row["canonical_position"]: row for row in rows}
    assert by_position[1]["protected"] is True
    assert by_position[1]["protection_reasons"] == ["motif_anchor"]
    assert by_position[2]["protected"] is True
    assert by_position[2]["protection_reasons"] == ["wang_ec86_direct_contact_prior"]
    assert by_position[3]["protected"] is False
    assert by_position[3]["non_fixed_missing_backbone"] is True
    assert by_position[4]["protected"] is True
    assert by_position[4]["protection_reasons"] == ["direct_retained_dna_rna_contact_5a"]
    assert by_position[5]["non_fixed"] is True
    assert by_position[6]["non_fixed"] is True
    assert by_position[6]["conservation_profile_ids"] == []
    assert by_position[7]["protected"] is True
    assert by_position[7]["protection_reasons"] == ["evolutionarily_conserved_clade9_25pct_plurality"]
    assert by_position[7]["conservation_profile_ids"] == ["ec86_clade9_conservation_v1"]
    assert summarize_mask_rows(rows)["source_protected_counts"] == {
        "motif_anchor": 1,
        "wang_ec86_direct_contact_prior": 1,
        "evolutionarily_conserved_clade9_25pct_plurality": 1,
        "direct_retained_dna_rna_contact_5a": 1,
    }


def _residue(position: int, status: str) -> dict[str, object]:
    return {
        "canonical_position": position,
        "wt_aa": "A",
        "structure_chain_id": "A",
        "structure_residue_id": position,
        "design_position": position,
        "mapping_status": status,
    }


def _contact_geometry(position: int, *, distance: float | None) -> dict[str, object]:
    return {
        "canonical_position": position,
        "nearest_context_atom_distance_angstrom": distance,
    }


def _conservation(position: int, *, profile_id: str, passes: bool) -> dict[str, object]:
    return {
        "canonical_position": position,
        "profile_id": profile_id,
        "passes_conservation_mask": passes,
    }
