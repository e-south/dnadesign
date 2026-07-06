"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/sequence_export.py

Selected RT protein-sequence export for Eco1 panel handoff planning.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from .constants import CODON_POLICY_ID

HANDOFF_SEQUENCE_CSV_FIELDS = [
    "candidate_id",
    "selection_slot",
    "design_class_id",
    "sequence_scope",
    "protein_sequence",
    "sequence_hash",
    "amino_acid_length",
    "protein_sequence_sha256",
    "mapped_rt_chain_length",
    "canonical_rt_length",
    "canonical_sequence_status",
    "canonical_sequence_sha256",
    "fold_review_class",
    "feasibility_status",
    "eligible_for_handoff",
    "codon_policy_id",
    "dna_design_status",
    "dna_sequence_status",
    "codon_optimization_status",
    "restriction_site_screen_status",
    "handoff_scope_note",
    "source_candidate_pool_sha256",
    "source_panel_sha256",
]


def write_candidate_handoff_sequence_csv(
    path: Path,
    *,
    panel_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    source_candidate_pool_sha256: str,
    source_panel_sha256: str,
) -> list[dict[str, object]]:
    """Write a flat selected-protein sequence table for review and handoff planning."""

    output_rows = build_candidate_handoff_sequence_rows(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        source_candidate_pool_sha256=source_candidate_pool_sha256,
        source_panel_sha256=source_panel_sha256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HANDOFF_SEQUENCE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def build_candidate_handoff_sequence_rows(
    *,
    panel_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    source_candidate_pool_sha256: str,
    source_panel_sha256: str,
) -> list[dict[str, object]]:
    """Return selected mapped RT-chain protein sequence rows with explicit non-DNA status."""

    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows}
    output_rows: list[dict[str, object]] = []
    for panel_row in panel_rows:
        candidate_id = str(panel_row["candidate_id"])
        candidate_row = candidate_by_id.get(candidate_id)
        if candidate_row is None:
            raise ValueError(f"Selected panel candidate is absent from candidate pool: {candidate_id}")
        sequence = str(candidate_row.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError(f"Selected panel candidate has no protein sequence: {candidate_id}")
        candidate_hash = str(candidate_row.get("sequence_hash") or "")
        panel_hash = str(panel_row.get("sequence_hash") or "")
        if candidate_hash != panel_hash:
            raise ValueError(
                "Selected panel sequence hash does not match candidate pool for "
                f"{candidate_id}: panel={panel_hash!r} candidate_pool={candidate_hash!r}"
            )
        output_rows.append(
            {
                "candidate_id": candidate_id,
                "selection_slot": str(panel_row.get("selection_slot") or ""),
                "design_class_id": str(panel_row.get("design_class_id") or ""),
                "sequence_scope": "mapped_rt_chain_protein",
                "protein_sequence": sequence,
                "sequence_hash": candidate_hash,
                "amino_acid_length": len(sequence),
                "protein_sequence_sha256": sequence_sha256(sequence),
                "mapped_rt_chain_length": len(sequence),
                "canonical_rt_length": 320,
                "canonical_sequence_status": "not_exported_in_this_slice",
                "canonical_sequence_sha256": "",
                "fold_review_class": str(panel_row.get("fold_review_class") or ""),
                "feasibility_status": str(panel_row.get("feasibility_status") or ""),
                "eligible_for_handoff": str(bool(panel_row.get("eligible_for_handoff"))).lower(),
                "codon_policy_id": CODON_POLICY_ID,
                "dna_design_status": "not_materialized",
                "dna_sequence_status": "not_dna",
                "codon_optimization_status": "not_codon_optimized",
                "restriction_site_screen_status": "not_screened",
                "handoff_scope_note": (
                    "RT protein sequence only; not DNA, codon optimized, restriction screened, or construct ready."
                ),
                "source_candidate_pool_sha256": source_candidate_pool_sha256,
                "source_panel_sha256": source_panel_sha256,
            }
        )
    return output_rows


def sequence_sha256(sequence: str) -> str:
    """Return the stable sequence hash used in the handoff sequence export."""

    return "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest()
