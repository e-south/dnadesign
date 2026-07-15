"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/foldcheck_structure_set_fixtures.py

Foldcheck structure-set fixtures for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    sha256,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.structure_file_fixtures import (
    write_mmcif_all_atom_reference,
    write_pdb,
)


def write_foldcheck_full_structure_set(review_root: Path) -> None:
    """Write compact local structure paths for foldcheck-linked review fixtures."""

    structure_root = review_root / "structures" / "full_fold_set"
    structure_root.mkdir(parents=True, exist_ok=True)
    reference_path = review_root / "structures" / "ec86kit_chain_a_backbone_reference.pdb"
    write_pdb(reference_path, residue_count=309)
    write_mmcif_all_atom_reference(review_root / "structures" / "ec86kit_protomer1_all_atom_reference.cif")
    write_pdb(structure_root / "wild_type.pdb", residue_count=311, coordinate_offset=12.0, include_sidechains=True)
    write_pdb(
        structure_root / "thread_candidate_alpha.pdb",
        residue_count=311,
        coordinate_offset=18.0,
        include_sidechains=True,
    )
    write_pdb(
        structure_root / "thread_candidate_beta.pdb",
        residue_count=311,
        coordinate_offset=24.0,
        include_sidechains=True,
    )
    structure_paths = {
        "wild_type": structure_root / "wild_type.pdb",
        "thread_candidate_alpha": structure_root / "thread_candidate_alpha.pdb",
        "thread_candidate_beta": structure_root / "thread_candidate_beta.pdb",
    }
    review_root.joinpath("foldcheck_full_structure_set.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt.foldcheck_full_structure_set",
                "schema_version": 1,
                "status": "materialized",
                "path_policy": "local_paths_manifest_relative",
                "source_request_hash": "sha256:" + "8" * 64,
                "structure_count": 3,
                "structures": [
                    {
                        "candidate_id": "wild_type",
                        "local_model_artifact_path": "structures/full_fold_set/wild_type.pdb",
                        "source_model_artifact_hash": "sha256:" + sha256(structure_paths["wild_type"]),
                        "copy_status": "already_local",
                        "display_label": "WT ColabFold baseline",
                        "full_sequence_identity_percent": 100.0,
                        "design_position_recovery_percent": None,
                        "proteinmpnn_rank": None,
                        "wt_runtime_ca_rmsd": None,
                    },
                    {
                        "candidate_id": "thread_candidate_alpha",
                        "local_model_artifact_path": "structures/full_fold_set/thread_candidate_alpha.pdb",
                        "source_model_artifact_hash": "sha256:" + sha256(structure_paths["thread_candidate_alpha"]),
                        "copy_status": "already_local",
                        "display_label": "ProteinMPNN variant rank 1",
                        "full_sequence_identity_percent": 99.375,
                        "design_position_recovery_percent": 72.0,
                        "proteinmpnn_rank": 1,
                        "wt_runtime_ca_rmsd": 0.82,
                    },
                    {
                        "candidate_id": "thread_candidate_beta",
                        "local_model_artifact_path": "structures/full_fold_set/thread_candidate_beta.pdb",
                        "source_model_artifact_hash": "sha256:" + sha256(structure_paths["thread_candidate_beta"]),
                        "copy_status": "already_local",
                        "display_label": "ProteinMPNN variant rank 2",
                        "full_sequence_identity_percent": 99.0625,
                        "design_position_recovery_percent": 55.0,
                        "proteinmpnn_rank": 2,
                        "wt_runtime_ca_rmsd": 3.12,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
