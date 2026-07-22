"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/selection_sequence_fixtures.py

Selected-panel sequence CSV fixtures for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def write_handoff_sequence_csv(path: Path, panel_rows: list[dict[str, object]]) -> None:
    sequences = {
        "thread_candidate_alpha": "MKSAGG",
        "thread_candidate_beta": "MKSGGG",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "selection_slot",
                "policy_id",
                "protein_sequence",
                "mapped_protein_sequence",
                "sequence_hash",
                "sequence_scope",
                "amino_acid_length",
                "protein_sequence_sha256",
                "mapped_protein_sequence_sha256",
                "mapped_rt_chain_length",
                "canonical_rt_length",
                "canonical_sequence_status",
                "canonical_sequence_sha256",
                "canonical_mutations",
                "fold_review_class",
                "eligible_for_handoff",
                "codon_policy_id",
                "dna_design_status",
                "dna_sequence_status",
                "codon_optimization_status",
                "restriction_site_screen_status",
                "handoff_scope_note",
                "source_candidate_pool_sha256",
                "source_panel_sha256",
                "source_foldcheck_input_sequences_sha256",
            ],
        )
        writer.writeheader()
        for row in panel_rows:
            mapped_sequence = sequences[str(row["candidate_id"])] + "A" * 303
            sequence = "AA" + mapped_sequence + "A" * 9
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "selection_slot": row["selection_slot"],
                    "policy_id": row["policy_id"],
                    "protein_sequence": sequence,
                    "mapped_protein_sequence": mapped_sequence,
                    "sequence_hash": f"fixture-{row['candidate_id']}",
                    "sequence_scope": "canonical_rt_protein",
                    "amino_acid_length": len(sequence),
                    "protein_sequence_sha256": "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                    "mapped_protein_sequence_sha256": "sha256:"
                    + hashlib.sha256(mapped_sequence.encode("utf-8")).hexdigest(),
                    "mapped_rt_chain_length": len(mapped_sequence),
                    "canonical_rt_length": 320,
                    "canonical_sequence_status": "materialized",
                    "canonical_sequence_sha256": "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                    "canonical_mutations": "A3G",
                    "fold_review_class": row["fold_review_class"],
                    "eligible_for_handoff": "true",
                    "codon_policy_id": "protein_sequence_only_no_codon_design_v1",
                    "dna_design_status": "not_materialized",
                    "dna_sequence_status": "not_dna",
                    "codon_optimization_status": "not_codon_optimized",
                    "restriction_site_screen_status": "not_screened",
                    "handoff_scope_note": "RT protein sequence only; not DNA, codon, restriction, or construct ready.",
                    "source_candidate_pool_sha256": "sha256:" + "a" * 64,
                    "source_panel_sha256": "sha256:" + "b" * 64,
                    "source_foldcheck_input_sequences_sha256": "sha256:" + "c" * 64,
                }
            )


__all__ = ["write_handoff_sequence_csv"]
