"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/foldcheck_fixtures.py

Fold-review fixtures for Eco1 review-deliverable materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.structure_file_fixtures import (
    write_mmcif_all_atom_reference,
    write_pdb,
)


def write_foldcheck_review_manifest(review_root: Path) -> None:
    """Write a compact fold-review manifest and ranking table."""

    plot_root = review_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    _write_foldcheck_candidate_ranking(review_root / "foldcheck_candidate_ranking.parquet")
    _write_foldcheck_full_structure_set(review_root)
    plot_root.joinpath("fold_metric_scatter.svg").write_text(
        '<svg role="img"><title>Fold metrics</title><desc>Fixture fold metrics.</desc></svg>\n',
        encoding="utf-8",
    )
    plot_root.joinpath("biohub_esmc_sae_coverage.svg").write_text(
        '<svg role="img"><title>SAE coverage</title><desc>Fixture SAE coverage.</desc></svg>\n',
        encoding="utf-8",
    )
    plot_root.joinpath("structure_overlay_panel.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    review_root.joinpath("review_visual_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt.foldcheck_review_visual_manifest",
                "status": "materialized",
                "plot_count": 4,
                "plots": [
                    {
                        "plot_id": "fold_metric_scatter",
                        "status": "rendered",
                        "path": "plots/fold_metric_scatter.svg",
                        "title": "ColabFold fold-check metric scatter",
                        "alt_text": "Fixture fold metric scatter.",
                        "description": "Fixture fold-review plot.",
                        "interpretation_limit": "Fold metrics do not measure activity.",
                        "data_sources": ["foldcheck_review/foldcheck_candidate_ranking.parquet"],
                        "skip_reason": "",
                    },
                    {
                        "plot_id": "biohub_esmc_sae_coverage",
                        "status": "rendered",
                        "path": "plots/biohub_esmc_sae_coverage.svg",
                        "title": "Biohub ESMC SAE coverage is complete for fold-reviewed sequences",
                        "alt_text": "Fixture Biohub ESMC SAE coverage plot.",
                        "description": "Fixture Biohub ESMC SAE coverage plot.",
                        "interpretation_limit": "SAE coverage is annotation coverage, not activity evidence.",
                        "data_sources": ["biohub_esmc_sae_profile.parquet"],
                        "skip_reason": "",
                    },
                    {
                        "plot_id": "structure_overlay_panel",
                        "status": "rendered",
                        "path": "plots/structure_overlay_panel.png",
                        "title": "Selected ColabFold structures align to the cryoEM reference",
                        "alt_text": "Fixture ChimeraX overlay panel.",
                        "description": "Fixture fold-review ChimeraX render.",
                        "interpretation_limit": "Structure overlays are visual review aids.",
                        "data_sources": [
                            "foldcheck_review/foldcheck_structure_panel.yaml",
                            "foldcheck_review/foldcheck_candidate_ranking.parquet",
                        ],
                        "skip_reason": "",
                    },
                    {
                        "plot_id": "structure_overlay_skipped",
                        "status": "skipped_runtime_unavailable",
                        "path": "plots/structure_overlay_skipped.png",
                        "title": "Skipped structure overlay fixture",
                        "alt_text": "Skipped overlay fixture.",
                        "description": "Fixture skipped fold-review ChimeraX render.",
                        "interpretation_limit": "Skipped renders do not support review.",
                        "data_sources": ["foldcheck_review/foldcheck_structure_panel.yaml"],
                        "skip_reason": "ChimeraX unavailable in fixture.",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_foldcheck_candidate_ranking(path: Path) -> None:
    rows = [
        {
            "candidate_id": "thread_candidate_alpha",
            "review_rank": 1,
            "plddt": 92.4,
            "wt_runtime_ca_rmsd": 0.82,
            "cryoem_mapped_ca_rmsd": 1.23,
            "seq_recovery": 0.72,
            "mutation_count": 2,
            "review_class": "strong_fold_preserved",
        },
        {
            "candidate_id": "thread_candidate_beta",
            "review_rank": 2,
            "plddt": 89.7,
            "wt_runtime_ca_rmsd": 3.12,
            "cryoem_mapped_ca_rmsd": 2.45,
            "seq_recovery": 0.55,
            "mutation_count": 3,
            "review_class": "review_band",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_foldcheck_full_structure_set(review_root: Path) -> None:
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
                        "copy_status": "already_local",
                        "display_label": "WT ColabFold baseline",
                        "sequence_identity_percent": 100.0,
                        "proteinmpnn_rank": None,
                        "wt_runtime_ca_rmsd": None,
                    },
                    {
                        "candidate_id": "thread_candidate_alpha",
                        "local_model_artifact_path": "structures/full_fold_set/thread_candidate_alpha.pdb",
                        "copy_status": "already_local",
                        "display_label": "ProteinMPNN variant rank 1",
                        "sequence_identity_percent": 72.0,
                        "proteinmpnn_rank": 1,
                        "wt_runtime_ca_rmsd": 0.82,
                    },
                    {
                        "candidate_id": "thread_candidate_beta",
                        "local_model_artifact_path": "structures/full_fold_set/thread_candidate_beta.pdb",
                        "copy_status": "already_local",
                        "display_label": "ProteinMPNN variant rank 2",
                        "sequence_identity_percent": 55.0,
                        "proteinmpnn_rank": 2,
                        "wt_runtime_ca_rmsd": 3.12,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
