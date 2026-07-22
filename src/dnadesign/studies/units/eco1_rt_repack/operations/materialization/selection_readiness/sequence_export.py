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
    "variant_id",
    "candidate_id",
    "selection_slot",
    "selection_rank",
    "policy_id",
    "design_group_id",
    "within_group_rank",
    "sequence_scope",
    "protein_sequence",
    "mapped_protein_sequence",
    "sequence_hash",
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
]

CANONICAL_RT_LENGTH = 320
MAPPED_RT_START = 3
MAPPED_RT_END = 311


def write_candidate_handoff_sequence_csv(
    path: Path,
    *,
    panel_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    canonical_sequences_by_id: Mapping[str, str],
    source_candidate_pool_sha256: str,
    source_panel_sha256: str,
    source_foldcheck_input_sequences_sha256: str,
) -> list[dict[str, object]]:
    """Write a flat selected-protein sequence table for review and handoff planning."""

    output_rows = build_candidate_handoff_sequence_rows(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        canonical_sequences_by_id=canonical_sequences_by_id,
        source_candidate_pool_sha256=source_candidate_pool_sha256,
        source_panel_sha256=source_panel_sha256,
        source_foldcheck_input_sequences_sha256=source_foldcheck_input_sequences_sha256,
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
    canonical_sequences_by_id: Mapping[str, str],
    source_candidate_pool_sha256: str,
    source_panel_sha256: str,
    source_foldcheck_input_sequences_sha256: str,
) -> list[dict[str, object]]:
    """Return selected canonical RT protein sequences with mapped-source provenance."""

    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows}
    output_rows: list[dict[str, object]] = []
    for panel_row in panel_rows:
        candidate_id = str(panel_row["candidate_id"])
        candidate_row = candidate_by_id.get(candidate_id)
        if candidate_row is None:
            raise ValueError(f"Selected panel candidate is absent from candidate pool: {candidate_id}")
        mapped_sequence = str(candidate_row.get("sequence") or "").strip().upper()
        if not mapped_sequence:
            raise ValueError(f"Selected panel candidate has no protein sequence: {candidate_id}")
        canonical_sequence = str(canonical_sequences_by_id.get(candidate_id) or "").strip().upper()
        if len(canonical_sequence) != CANONICAL_RT_LENGTH:
            raise ValueError(
                f"Selected panel candidate {candidate_id} lacks a {CANONICAL_RT_LENGTH}-aa canonical fold input"
            )
        canonical_mapped_sequence = canonical_sequence[MAPPED_RT_START - 1 : MAPPED_RT_END]
        if canonical_mapped_sequence != mapped_sequence:
            raise ValueError(
                f"Canonical fold input does not match mapped candidate sequence for {candidate_id}: "
                f"canonical positions {MAPPED_RT_START}-{MAPPED_RT_END} differ"
            )
        candidate_hash = str(candidate_row.get("sequence_hash") or "")
        panel_hash = str(panel_row.get("sequence_hash") or "")
        if candidate_hash != panel_hash:
            raise ValueError(
                "Selected panel sequence hash does not match candidate pool for "
                f"{candidate_id}: panel={panel_hash!r} candidate_pool={candidate_hash!r}"
            )
        output_rows.append(
            {
                "variant_id": str(panel_row.get("variant_id") or ""),
                "candidate_id": candidate_id,
                "selection_slot": str(panel_row.get("selection_slot") or ""),
                "selection_rank": int(panel_row["selection_rank"]),
                "policy_id": str(panel_row.get("policy_id") or ""),
                "design_group_id": str(panel_row.get("design_group_id") or ""),
                "within_group_rank": int(panel_row["within_group_rank"]),
                "sequence_scope": "canonical_rt_protein",
                "protein_sequence": canonical_sequence,
                "mapped_protein_sequence": mapped_sequence,
                "sequence_hash": candidate_hash,
                "amino_acid_length": len(canonical_sequence),
                "protein_sequence_sha256": sequence_sha256(canonical_sequence),
                "mapped_protein_sequence_sha256": sequence_sha256(mapped_sequence),
                "mapped_rt_chain_length": len(mapped_sequence),
                "canonical_rt_length": CANONICAL_RT_LENGTH,
                "canonical_sequence_status": "materialized",
                "canonical_sequence_sha256": sequence_sha256(canonical_sequence),
                "canonical_mutations": ";".join(str(value) for value in candidate_row.get("canonical_mutations") or []),
                "fold_review_class": str(panel_row.get("fold_review_class") or ""),
                "eligible_for_handoff": str(bool(panel_row.get("eligible_for_handoff"))).lower(),
                "codon_policy_id": CODON_POLICY_ID,
                "dna_design_status": "not_materialized",
                "dna_sequence_status": "not_dna",
                "codon_optimization_status": "not_codon_optimized",
                "restriction_site_screen_status": "not_screened",
                "handoff_scope_note": (
                    "Canonical 320-aa RT protein sequence only; not DNA, codon optimized, restriction screened, "
                    "or construct ready."
                ),
                "source_candidate_pool_sha256": source_candidate_pool_sha256,
                "source_panel_sha256": source_panel_sha256,
                "source_foldcheck_input_sequences_sha256": source_foldcheck_input_sequences_sha256,
            }
        )
    return output_rows


def sequence_sha256(sequence: str) -> str:
    """Return the stable sequence hash used in the handoff sequence export."""

    return "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def read_fasta_sequences(path: Path) -> dict[str, str]:
    """Read uppercase FASTA sequences keyed by the first header token."""

    sequences: dict[str, list[str]] = {}
    current_id = ""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            current_id = stripped[1:].split()[0].strip()
            if not current_id:
                raise ValueError(f"FASTA header without sequence id at {path}:{line_number}")
            if current_id in sequences:
                raise ValueError(f"Duplicate FASTA sequence id {current_id!r} at {path}:{line_number}")
            sequences[current_id] = []
            continue
        if not current_id:
            raise ValueError(f"FASTA sequence appears before a header at {path}:{line_number}")
        sequences[current_id].append(stripped.upper())
    return {candidate_id: "".join(parts) for candidate_id, parts in sequences.items()}
