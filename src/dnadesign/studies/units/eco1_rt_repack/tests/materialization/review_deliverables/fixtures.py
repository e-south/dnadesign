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
from dnadesign.thread.candidates.proteinmpnn import write_candidate_table
from dnadesign.thread.foldcheck import sequence_hash

from .biohub_sae_fixtures import write_biohub_esmc_sae_outputs
from .conservation_fixtures import write_conservation_inputs
from .esmc_fixtures import write_wt_mutation_scoring_outputs
from .foldcheck_fixtures import write_foldcheck_review_manifest
from .selection_fixtures import write_selection_readiness_manifest


def write_deliverable_inputs(output_root: Path) -> None:
    """Write a compact Eco1-like artifact set for review-deliverable tests."""

    output_root.mkdir(parents=True, exist_ok=True)
    write_conservation_inputs(output_root)
    _write_mask_set(output_root / "mask_set.yaml")
    _write_design_class_mask_sets(output_root / "design_classes")
    _write_candidate_table(output_root / "candidate_table.parquet")
    _write_candidate_table(output_root / "design_classes" / "candidate_pool.parquet", include_design_classes=True)
    _write_reference_pdb(output_root / "proteinmpnn_request" / "chain_a_backbone.pdb")
    write_foldcheck_review_manifest(output_root / "foldcheck_review")
    write_foldcheck_review_manifest(output_root / "design_classes" / "foldcheck_review")
    write_wt_mutation_scoring_outputs(output_root)
    write_biohub_esmc_sae_outputs(output_root)
    write_selection_readiness_manifest(output_root / "design_classes" / "selection")


def write_rt_annotation_context_sources(output_root: Path) -> tuple[Path, Path]:
    """Write compact RT annotation context sources for visual-rendering tests."""

    annotation_tracks_path = output_root / "rt-annotation-tracks.yaml"
    manual_authority_path = output_root / "manual-mask-authority.yaml"
    target_hash = "sha256:" + "a" * 64
    annotation_tracks_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "dnadesign.aligner.msa.visualization.annotation_tracks",
                "schema_version": 1,
                "study_id": "eco1_rt_repack",
                "status": "fixture_rt_interval_authority_v1",
                "coordinate_space": "target_ungapped_position",
                "target_row_id": "eco1_rt_ec86kit_reference",
                "target_sequence_hash": target_hash,
                "source_basis": [],
                "tracks": [
                    {
                        "id": "retron_rt_context_spans",
                        "label": "Mask-context spans",
                        "features": [
                            {
                                "id": "retron_x_context",
                                "label": "Region X local context",
                                "start": 2,
                                "end": 4,
                            },
                            {
                                "id": "catalytic_context",
                                "label": "Catalytic YADD local context",
                                "start": 3,
                                "end": 5,
                            },
                        ],
                    },
                    {
                        "id": "retron_rt_core_intervals",
                        "label": "RT1-RT7 core intervals",
                        "features": [
                            {
                                "id": "rt1_interval",
                                "label": "RT1",
                                "start": 2,
                                "end": 3,
                            },
                            {
                                "id": "rt2_interval",
                                "label": "RT2",
                                "start": 4,
                                "end": 5,
                            },
                        ],
                    },
                    {
                        "id": "retron_rt_motif_anchors",
                        "label": "RT motif anchors",
                        "features": [
                            {
                                "id": "retron_x_naxxh",
                                "label": "NAxxH",
                                "start": 3,
                                "end": 3,
                            },
                            {
                                "id": "catalytic_yadd",
                                "label": "YADD",
                                "start": 4,
                                "end": 4,
                            },
                        ],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manual_authority_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt_repack.manual_mask_authority_source",
                "schema_version": 1,
                "study_id": "eco1_rt_repack",
                "status": "fixture_motif_review_labels_v1",
                "coordinate_space": "canonical_position",
                "target_row_id": "eco1_rt_ec86kit_reference",
                "target_sequence_hash": target_hash,
                "mask_policy_id": "eco1_rt_manual_motif_wang_direct_contact_v1",
                "source_basis": [],
                "authority_sets": [
                    _authority_set(
                        "ec86_rt1_interval", "rt_core_interval", "review_label", "rt1_interval", "RT1", 2, 3
                    ),
                    _authority_set(
                        "ec86_rt2_interval", "rt_core_interval", "review_label", "rt2_interval", "RT2", 4, 5
                    ),
                    _authority_set(
                        "ec86_retron_x_region",
                        "retron_x_motif_anchor",
                        "fixed",
                        "retron_x_naxxh",
                        "NAxxH",
                        3,
                        3,
                    ),
                    _authority_set(
                        "ec86_active_site_geometry",
                        "catalytic_core_motif_anchor",
                        "fixed",
                        "catalytic_yadd",
                        "YADD",
                        4,
                        4,
                    ),
                ],
                "context_only_spans": [
                    {
                        "id": "retron_x_context",
                        "label": "Region X local context",
                        "start": 2,
                        "end": 4,
                    },
                    {
                        "id": "catalytic_context",
                        "label": "Catalytic YADD local context",
                        "start": 3,
                        "end": 5,
                    },
                ],
                "deferred_authority": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return annotation_tracks_path, manual_authority_path


def _authority_set(
    set_id: str,
    authority_type: str,
    policy: str,
    feature_id: str,
    label: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    return {
        "id": set_id,
        "label": label,
        "authority_type": authority_type,
        "policy": policy,
        "features": [
            {
                "id": feature_id,
                "label": label,
                "start": start,
                "end": end,
                "reason": feature_id,
                "source_locator": "fixture",
                "evidence_basis": ["fixture"],
            }
        ],
    }


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


def _write_candidate_table(path: Path, *, include_design_classes: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    design_class_ids = [spec.design_class_id for spec in ALL_SPECS]
    for rank, (candidate_id, score, global_score, seq_recovery, temperature, mutation_count) in enumerate(
        [
            ("thread_candidate_alpha", 1.1, 1.6, 0.72, 0.1, 2),
            ("thread_candidate_beta", 1.4, 1.9, 0.55, 0.3, 3),
        ],
        start=1,
    ):
        sequence = "MKSAYL"[: 6 - mutation_count] + "G" * mutation_count
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_sample_id": f"sample-{rank}",
                "backend_run_id": "proteinmpnn-fixture",
                "request_hash": "sha256:" + "4" * 64,
                "sequence_hash": sequence_hash(sequence),
                "sequence": sequence,
                "score": score,
                "global_score": global_score,
                "seq_recovery": seq_recovery,
                "seed": 101,
                "temperature": temperature,
                "sample_index": rank,
                "duplicate_sample_count": 1,
                "mutation_count": mutation_count,
                "mutable_mutation_count": mutation_count,
                "protected_mutation_count": 0,
                "outside_mutable_positions": [],
                "canonical_mutations": [f"A{position}G" for position in range(1, mutation_count + 1)],
                "status": "accepted",
                "rank": rank,
            }
        )
        if include_design_classes:
            rows[-1]["design_class_id"] = design_class_ids[rank - 1]
            rows[-1]["mask_policy_id"] = design_class_ids[rank - 1]
            rows[-1]["class_priority"] = rank - 1
    write_candidate_table(path, rows, request_hash="sha256:" + "4" * 64)


def _write_reference_pdb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(1, 7):
        atom_prefix = f"ATOM  {index:5d}  CA  ALA A{index:4d}"
        coords = f"{float(index):8.3f}{0.0:8.3f}{0.0:8.3f}"
        atom_suffix = "  1.00  0.00           C"
        lines.append(f"{atom_prefix}    {coords}{atom_suffix}")
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
