"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/twist_handoff/_fixtures.py

Input fixtures for Eco1 RT Twist handoff materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    variant_id_for_policy,
)
from dnadesign.thread.foldcheck import sequence_hash

REPO_ROOT = Path(__file__).resolve().parents[8]
WT_GENBANK = Path("docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/genbank/retron-eco1-rt.gb")


def write_twist_handoff_inputs(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    wt_record = SeqIO.read(REPO_ROOT / WT_GENBANK, "genbank")
    wt_protein = str(wt_record.seq.translate())[:-1]
    policy_ids = (
        "distal_scaffold_repack_v1",
        "distal_scaffold_repack_v1",
        "near_dna_rna_acid_free_v1",
        "near_dna_rna_acid_free_v1",
        "combined_near_acid_free_plus_distal_v1",
        "combined_near_acid_free_plus_distal_v1",
        "near_dna_rna_acid_free_v1",
        "combined_near_acid_free_plus_distal_v1",
    )
    panel_rows: list[dict[str, object]] = []
    pool_rows: list[dict[str, object]] = []
    fasta_records: list[str] = [f">wild_type\n{wt_protein}\n"]
    within_group_ranks = (1, 2, 1, 2, 1, 2, 3, 3)
    for index, (policy_id, within_group_rank) in enumerate(zip(policy_ids, within_group_ranks, strict=True), start=1):
        position = index + 2
        ref = wt_protein[position - 1]
        alt = "G" if ref != "G" else "A"
        protein = wt_protein[: position - 1] + alt + wt_protein[position:]
        candidate_id = f"candidate_{index}"
        mapped_sequence = protein[2:311]
        mapped_hash = sequence_hash(mapped_sequence)
        token = f"{ref}{position}{alt}"
        panel_rows.append(
            {
                "variant_id": variant_id_for_policy(
                    policy_id=policy_id,
                    within_group_rank=within_group_rank,
                ),
                "candidate_id": candidate_id,
                "selection_slot": f"slot_{index}",
                "selection_rank": index,
                "sequence_hash": mapped_hash,
                "policy_id": policy_id,
                "design_group_id": (
                    "distal_scaffold_repack"
                    if policy_id == "distal_scaffold_repack_v1"
                    else "peripheral_shell_repack"
                    if policy_id == "near_dna_rna_acid_free_v1"
                    else "combined_peripheral_and_distal_repack"
                ),
                "within_group_rank": within_group_rank,
                "wang_alpha1_r13_review_status": "retained_wt",
                "wang_alpha1_mutation_count": int(4 <= position <= 16),
                "wang_alpha1_f10_substitution": token if position == 10 else "WT",
                "wang_alpha1_r13_substitution": token if position == 13 else "WT",
                "wang_r13a_interface_disruption_evidence_match": token == "R13A",
                "rt_msdna_oligomeric_state_review_status": "not_established",
                "eligible_for_handoff": True,
            }
        )
        pool_rows.append(
            {
                "candidate_id": candidate_id,
                "sequence_hash": mapped_hash,
                "sequence": mapped_sequence,
                "canonical_mutations": [token],
            }
        )
        fasta_records.append(f">{candidate_id}\n{protein}\n")
    policy_rows = [
        {
            "policy_id": policy_id,
            "eco1_position": position,
            "protected_reason_codes": ["motif_context"] if position == 100 else [],
            "is_direct_contact_le_5a": False,
            "is_near_region_gt5_le10a": False,
            "is_wang_thumb_track": False,
            "is_c_terminal_thumb_context": False,
            "is_conserved_core": position == 100,
        }
        for policy_id in sorted(set(policy_ids))
        for position in range(1, 321)
    ]
    panel_path = root / "candidate_selection_panel.parquet"
    pool_path = root / "candidate_pool.parquet"
    policy_path = root / "generation_policy_positions.parquet"
    fasta_path = root / "input_sequences.fasta"
    pq.write_table(pa.Table.from_pylist(panel_rows), panel_path)
    pq.write_table(pa.Table.from_pylist(pool_rows), pool_path)
    pq.write_table(pa.Table.from_pylist(policy_rows), policy_path)
    fasta_path.write_text("".join(fasta_records), encoding="utf-8")
    return {
        "candidate_selection_panel_path": panel_path,
        "candidate_pool_path": pool_path,
        "foldcheck_fasta_path": fasta_path,
        "generation_policy_positions_path": policy_path,
        "wild_type_genbank_path": REPO_ROOT / WT_GENBANK,
    }
