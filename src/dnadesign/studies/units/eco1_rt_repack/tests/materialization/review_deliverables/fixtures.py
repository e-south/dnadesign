"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/fixtures.py

Fixtures for Eco1 review-deliverable materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.constants import (
    BASELINE_CLASS_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)

from .biohub_sae_fixtures import write_biohub_esmc_sae_outputs
from .candidate_pool_fixtures import write_candidate_pool, write_generation_policy_positions, write_reference_pdb
from .conservation_fixtures import write_conservation_inputs
from .esmc_fixtures import write_wt_mutation_scoring_outputs
from .foldcheck_fixtures import write_foldcheck_review_manifest
from .rt_annotation_fixtures import write_rt_annotation_context_sources as write_rt_annotation_context_sources
from .selection_fixtures import write_selection_readiness_manifest


def write_deliverable_inputs(output_root: Path) -> None:
    """Write a compact Eco1-like artifact set for review-deliverable tests."""

    output_root.mkdir(parents=True, exist_ok=True)
    write_conservation_inputs(output_root)
    _write_mask_set(output_root / "mask_set.yaml")
    _write_design_class_mask_sets(output_root / "design_classes")
    write_candidate_pool(output_root / "candidate_table.parquet")
    write_candidate_pool(output_root / "design_classes" / "candidate_pool.parquet", include_design_classes=True)
    write_candidate_pool(
        output_root / "generation_policies_v3" / "candidate_pool.parquet",
        include_generation_policies=True,
    )
    write_generation_policy_positions(output_root / "generation_policies_v3" / "generation_policy_positions.parquet")
    write_reference_pdb(output_root / "proteinmpnn_request" / "chain_a_backbone.pdb")
    write_foldcheck_review_manifest(output_root / "foldcheck_review")
    write_foldcheck_review_manifest(output_root / "design_classes" / "foldcheck_review")
    write_foldcheck_review_manifest(output_root / "generation_policies_v3" / "foldcheck_review")
    write_wt_mutation_scoring_outputs(output_root)
    write_biohub_esmc_sae_outputs(output_root)
    write_selection_readiness_manifest(output_root / "design_classes" / "selection")
    write_selection_readiness_manifest(output_root / "generation_policies_v3" / "selection")


def _write_mask_set(
    path: Path,
    *,
    mask_policy_id: str = BASELINE_CLASS_ID,
    conservation_profile_id: str = "ec86_clade9_conservation_v1",
    conservation_threshold: float = 0.25,
    contact_threshold_angstrom: float = 5.0,
    selected_conservation_positions: set[int] | None = None,
    selected_contact_positions: set[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_conservation_positions = selected_conservation_positions or {2, 4}
    selected_contact_positions = selected_contact_positions or {5}
    residues: list[dict[str, Any]] = []
    for position, wt_aa in enumerate("MKSAYL", start=1):
        selected_conservation = position in selected_conservation_positions
        selected_contact = position in selected_contact_positions
        protected = position == 3 or position == 4 or selected_conservation or selected_contact
        residues.append(
            {
                "canonical_position": position,
                "wt_aa": wt_aa,
                "motif_protected": position == 3,
                "wang_ec86_direct_contact_prior": position == 4,
                "direct_retained_dna_rna_contact_5a": position == 5,
                "selected_conservation_profile_id": conservation_profile_id,
                "selected_conservation_threshold": conservation_threshold,
                "selected_conservation_rule_passed": selected_conservation,
                "selected_contact_threshold_angstrom": contact_threshold_angstrom,
                "selected_retained_dna_rna_contact": selected_contact,
                "evolutionarily_conserved_clade9_25pct_plurality": position in {2, 4},
                "protected": protected,
                "non_fixed": not protected,
                "non_fixed_missing_backbone": False,
                "mapping_status": "mapped",
                "structure_chain_id": "A",
                "structure_residue_id": position,
                "has_backbone_coordinates": True,
                "manual_mask_reason": "retron_x_naxxh" if position == 3 else "",
                "rt_interval_review_label": "RT1" if position in {1, 2} else "",
                "protection_reasons": ["fixture"] if position in {2, 3, 4, 5} else [],
            }
        )
    path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "thread.mask_set",
                "schema_version": 1,
                "status": "materialized",
                "mask_policy_id": mask_policy_id,
                "summary": {
                    "design_class_id": mask_policy_id,
                    "selected_conservation_profile_id": conservation_profile_id,
                    "selected_conservation_threshold": conservation_threshold,
                    "selected_contact_threshold_angstrom": contact_threshold_angstrom,
                    "total_positions": 6,
                    "protected_position_count": sum(1 for row in residues if row["protected"]),
                    "non_fixed_mapped_position_count": sum(1 for row in residues if row["non_fixed"]),
                },
                "residues": residues,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_design_class_mask_sets(design_classes_root: Path) -> None:
    class_positions = {
        "eco1_rt_clade9_plurality25_contact6a_v1": ({2, 4}, {4, 5}),
        "eco1_rt_clade9_plurality25_contact8a_v1": ({2, 4}, {3, 4, 5}),
        "eco1_rt_clade9_plurality25_contact10a_v1": ({2, 4}, {2, 3, 4, 5}),
        "eco1_rt_clade9_plurality50_contact5a_v1": ({4}, {5}),
        "eco1_rt_iia3_cluster42_1_plurality50_contact5a_v1": ({1, 3, 5}, {5}),
    }
    for spec in ALL_SPECS:
        if spec.design_class_id == BASELINE_CLASS_ID:
            continue
        conservation_positions, contact_positions = class_positions[spec.design_class_id]
        _write_mask_set(
            design_classes_root / spec.design_class_id / "mask_set.yaml",
            mask_policy_id=spec.design_class_id,
            conservation_profile_id=spec.conservation_profile_id,
            conservation_threshold=spec.conservation_threshold,
            contact_threshold_angstrom=spec.contact_threshold_angstrom,
            selected_conservation_positions=conservation_positions,
            selected_contact_positions=contact_positions,
        )
