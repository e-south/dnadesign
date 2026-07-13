"""Candidate-pool and structure fixtures for Eco1 review-deliverable tests."""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.thread.candidates.proteinmpnn import write_candidate_table
from dnadesign.thread.foldcheck import sequence_hash


def write_generation_policy_positions(path: Path) -> None:
    """Write a compact fixed/open policy-position fixture."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    policy_open_positions = {
        DISTAL_SCAFFOLD_POLICY_ID: {1},
        NEAR_DNA_RNA_ACID_FREE_POLICY_ID: {6},
        COMBINED_NEAR_PLUS_DISTAL_POLICY_ID: {1, 6},
    }
    rows = []
    for policy_id, open_positions in policy_open_positions.items():
        for position, wt_aa in enumerate("MKSAYL", start=1):
            is_open = position in open_positions
            is_near = position == 6
            rows.append(
                {
                    "policy_id": policy_id,
                    "policy_version": 3,
                    "open_set_id": "fixture_open_set",
                    "eco1_position": position,
                    "wt_aa": wt_aa,
                    "structure_position": position,
                    "chain_position": position,
                    "is_mapped": True,
                    "is_designable_backbone_position": True,
                    "protected_reason_codes": [] if is_open else ["fixture_protected"],
                    "distance_to_retained_dna_rna": 7.0 if is_near else 14.0,
                    "is_direct_contact_le_5a": position == 5,
                    "is_near_region_gt5_le10a": is_near,
                    "is_wang_thumb_track": position == 4,
                    "is_c_terminal_thumb_context": False,
                    "is_conserved_core": position in {2, 4},
                    "motif_context_codes": ["fixture_motif"] if position == 3 else [],
                    "is_open_position": is_open,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def write_candidate_pool(
    path: Path,
    *,
    include_design_classes: bool = False,
    include_generation_policies: bool = False,
) -> None:
    """Write a compact ProteinMPNN candidate table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    design_class_ids = [spec.design_class_id for spec in ALL_SPECS]
    fixtures = [
        ("thread_candidate_alpha", 1.1, 1.6, 0.72, 0.1, 2),
        ("thread_candidate_beta", 1.4, 1.9, 0.55, 0.3, 3),
    ]
    for rank, (candidate_id, score, global_score, seq_recovery, temperature, mutation_count) in enumerate(
        fixtures,
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
        if include_generation_policies:
            policy_id = (DISTAL_SCAFFOLD_POLICY_ID, NEAR_DNA_RNA_ACID_FREE_POLICY_ID)[rank - 1]
            rows[-1]["policy_id"] = policy_id
            rows[-1]["policy_version"] = 3
            rows[-1]["primary_policy_id"] = policy_id
            rows[-1]["source_policy_ids"] = [policy_id]
    write_candidate_table(path, rows, request_hash="sha256:" + "4" * 64)


def write_reference_pdb(path: Path) -> None:
    """Write a six-residue all-alpha-carbon PDB fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(1, 7):
        atom_prefix = f"ATOM  {index:5d}  CA  ALA A{index:4d}"
        coords = f"{float(index):8.3f}{0.0:8.3f}{0.0:8.3f}"
        atom_suffix = "  1.00  0.00           C"
        lines.append(f"{atom_prefix}    {coords}{atom_suffix}")
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
