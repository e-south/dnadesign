"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/sae_window_summary/windows.py

Window definitions for Eco1 SAE local-feature review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.sae_window_summary.models import WindowSpec


def default_window_specs(mask_rows: list[dict[str, Any]]) -> tuple[WindowSpec, ...]:
    """Build the three Eco1 SAE review windows from generated mask rows."""

    catalytic = tuple(range(189, 212))
    contact_surface = tuple(
        sorted(
            int(row["canonical_position"])
            for row in mask_rows
            if row.get("mapping_status") == "mapped"
            and row.get("min_distance_to_retained_dna_rna_angstrom") is not None
            and float(row["min_distance_to_retained_dna_rna_angstrom"]) <= 5.0
        )
    )
    mutable_annulus_basic_surface = tuple(
        sorted(
            {
                int(row["canonical_position"])
                for row in mask_rows
                if row.get("mapping_status") == "mapped"
                and bool(row.get("non_fixed"))
                and row.get("min_distance_to_retained_dna_rna_angstrom") is not None
                and (
                    5.0 < float(row["min_distance_to_retained_dna_rna_angstrom"]) <= 12.0
                    or str(row.get("wt_aa") or "") in {"H", "K", "R"}
                )
            }
        )
    )
    specs = (
        WindowSpec(
            "catalytic_palm_control",
            "Catalytic palm control",
            catalytic,
            "YADD-centered palm neighborhood that should remain close to WT in the conservative design.",
        ),
        WindowSpec(
            "thumb_palm_na_binding_surface",
            "Nucleic-acid contact surface",
            contact_surface,
            "Mapped residues within 5 A of retained DNA/RNA in the Ec86 reference structure.",
        ),
        WindowSpec(
            "mutable_substrate_proximal_annulus_basic_surface",
            "Mutable near retained DNA/RNA and basic surface",
            mutable_annulus_basic_surface,
            (
                "Unprotected mapped residues outside the 5 A protected shell "
                "and within 12 A of retained DNA/RNA, plus unprotected WT H/K/R "
                "surface positions."
            ),
        ),
    )
    _validate_specs(specs)
    return specs


def _validate_specs(specs: tuple[WindowSpec, ...]) -> None:
    for spec in specs:
        if not spec.residue_positions_1based:
            raise ValueError(f"SAE window {spec.window_id!r} has no positions")
        if len(set(spec.residue_positions_1based)) != len(spec.residue_positions_1based):
            raise ValueError(f"SAE window {spec.window_id!r} has duplicate positions")
