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

from dnadesign.thread.candidates.proteinmpnn import write_candidate_table
from dnadesign.thread.foldcheck import sequence_hash

from .biohub_sae_fixtures import write_biohub_esmc_sae_outputs
from .conservation_fixtures import write_conservation_inputs
from .esmc_fixtures import write_wt_mutation_scoring_outputs
from .foldcheck_fixtures import write_foldcheck_review_manifest


def write_deliverable_inputs(output_root: Path) -> None:
    """Write a compact Eco1-like artifact set for review-deliverable tests."""

    output_root.mkdir(parents=True, exist_ok=True)
    write_conservation_inputs(output_root)
    _write_mask_set(output_root / "mask_set.yaml")
    _write_candidate_table(output_root / "candidate_table.parquet")
    _write_reference_pdb(output_root / "proteinmpnn_request" / "chain_a_backbone.pdb")
    write_foldcheck_review_manifest(output_root / "foldcheck_review")
    write_wt_mutation_scoring_outputs(output_root)
    write_biohub_esmc_sae_outputs(output_root)


def _write_mask_set(path: Path) -> None:
    residues: list[dict[str, Any]] = []
    for position, wt_aa in enumerate("MKSAYL", start=1):
        residues.append(
            {
                "canonical_position": position,
                "wt_aa": wt_aa,
                "motif_protected": position == 3,
                "wang_ec86_direct_contact_prior": position == 4,
                "direct_retained_dna_rna_contact_5a": position == 5,
                "evolutionarily_conserved_clade9_25pct_plurality": position in {2, 4},
                "protected": position in {2, 3, 4, 5},
                "non_fixed": position in {1, 6},
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
                "mask_policy_id": "eco1_rt_clade9_plurality25_direct_contact5a_v1",
                "summary": {
                    "total_positions": 6,
                    "protected_position_count": 4,
                    "non_fixed_mapped_position_count": 2,
                },
                "residues": residues,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_candidate_table(path: Path) -> None:
    rows = []
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
