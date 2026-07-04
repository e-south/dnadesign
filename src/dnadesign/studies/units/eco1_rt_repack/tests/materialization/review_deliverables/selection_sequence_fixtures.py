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
                "design_class_id",
                "protein_sequence",
                "sequence_hash",
                "amino_acid_length",
                "fold_review_class",
                "feasibility_status",
                "eligible_for_handoff",
                "codon_policy_id",
                "dna_design_status",
                "restriction_site_screen_status",
            ],
        )
        writer.writeheader()
        for row in panel_rows:
            sequence = sequences[str(row["candidate_id"])]
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "selection_slot": row["selection_slot"],
                    "design_class_id": row["design_class_id"],
                    "protein_sequence": sequence,
                    "sequence_hash": f"fixture-{row['candidate_id']}",
                    "amino_acid_length": len(sequence),
                    "fold_review_class": row["fold_review_class"],
                    "feasibility_status": row["feasibility_status"],
                    "eligible_for_handoff": "true",
                    "codon_policy_id": "protein_sequence_only_no_codon_design_v1",
                    "dna_design_status": "not_materialized",
                    "restriction_site_screen_status": "not_applicable_until_dna_sequence_materialized",
                }
            )


__all__ = ["write_handoff_sequence_csv"]
