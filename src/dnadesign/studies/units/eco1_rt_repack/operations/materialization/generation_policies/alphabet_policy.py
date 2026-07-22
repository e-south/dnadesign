"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/alphabet_policy.py

Per-residue amino-acid rules for Eco1 RT generation policies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from dnadesign.aligner.msa import load_fasta_records

from .constants import (
    ACIDIC_AMINO_ACIDS,
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    CONSERVATION_PROFILE_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    GENERATION_POLICY_VERSION,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
    PROLINE_GLYCINE_AMINO_ACIDS,
    PROTEINMPNN_ALPHABET,
    STANDARD_AMINO_ACIDS,
    STANDARD_AMINO_ACIDS_NO_CYS,
    TARGET_ALIGNMENT_ROW_ID,
)
from .models import GenerationPolicyConfig


def build_alphabet_rows(
    *,
    config: GenerationPolicyConfig,
    position_rows: list[dict[str, Any]],
    conservation_rows: list[dict[str, Any]],
    source_root: Path,
) -> list[dict[str, Any]]:
    """Build global distal and residue-specific proximal alphabet rows."""

    clade9_counts = _profile_residue_counts(
        profile_id=CONSERVATION_PROFILE_ID,
        conservation_rows=conservation_rows,
        alignment_path=source_root / "conservation_alignments" / f"{CONSERVATION_PROFILE_ID}.aligned.fasta",
    )
    rows: list[dict[str, Any]] = []
    for policy in config.enabled_policies:
        if policy.policy_id in (DISTAL_SCAFFOLD_POLICY_ID, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID):
            rows.append(_distal_alphabet_row(policy_id=policy.policy_id))
        if policy.policy_id in (NEAR_DNA_RNA_ACID_FREE_POLICY_ID, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID):
            rows.extend(
                _near_region_alphabet_rows(
                    policy_id=policy.policy_id,
                    position_rows=position_rows,
                    residue_counts_by_position=clade9_counts,
                )
            )
    return rows


def _distal_alphabet_row(*, policy_id: str) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "policy_version": GENERATION_POLICY_VERSION,
        "alphabet_scope": "distal_scaffold",
        "alphabet_rule_id": "broad_no_new_cysteine",
        "alphabet_enforcement_mode": "upstream_omit_AAs_C",
        "eco1_position": None,
        "wt_aa": None,
        "allowed_amino_acids": list(STANDARD_AMINO_ACIDS_NO_CYS),
        "disallowed_amino_acids": ["C"],
        "observed_amino_acids": [],
        "interpretation_limit": "The distal alphabet does not imply a substrate-facing chemistry claim.",
    }


def _near_region_alphabet_rows(
    *,
    policy_id: str,
    position_rows: list[dict[str, Any]],
    residue_counts_by_position: Mapping[int, Counter[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position_row in position_rows:
        if position_row["policy_id"] != policy_id:
            continue
        if not position_row["is_open_position"] or not position_row["is_near_region_gt5_le10a"]:
            continue
        position = int(position_row["eco1_position"])
        wt_aa = str(position_row["wt_aa"])
        observed = residue_counts_by_position.get(position, Counter())
        allowed = _allowed_near_region_amino_acids(wt_aa=wt_aa, observed=observed)
        rows.append(
            {
                "policy_id": policy_id,
                "policy_version": GENERATION_POLICY_VERSION,
                "alphabet_scope": "near_dna_rna_gt5_le10_excluding_protected",
                "alphabet_rule_id": "msa_observed_acid_free_basic_polar_neutral",
                "alphabet_enforcement_mode": "upstream_omit_AA_jsonl",
                "eco1_position": position,
                "wt_aa": wt_aa,
                "allowed_amino_acids": allowed,
                "disallowed_amino_acids": [aa for aa in PROTEINMPNN_ALPHABET if aa not in allowed],
                "observed_amino_acids": _ordered_amino_acids(aa for aa, count in observed.items() if count > 0),
                "interpretation_limit": (
                    "Near retained DNA/RNA alphabets allow only MSA-observed alternatives with no new acidic "
                    "residues. The v3 "
                    "global no-cysteine rule can force an open WT cysteine to change; the alphabet does not assert "
                    "that added basic charge improves function."
                ),
            }
        )
    return rows


def _allowed_near_region_amino_acids(*, wt_aa: str, observed: Counter[str]) -> list[str]:
    allowed = {wt_aa} if wt_aa in STANDARD_AMINO_ACIDS_NO_CYS else set()
    for aa, count in observed.items():
        if count <= 0:
            continue
        if aa not in STANDARD_AMINO_ACIDS_NO_CYS:
            continue
        if aa in ACIDIC_AMINO_ACIDS:
            continue
        if aa in PROLINE_GLYCINE_AMINO_ACIDS and aa != wt_aa:
            continue
        allowed.add(aa)
    if not allowed:
        raise ValueError(f"near-region alphabet has no allowed amino acids for WT {wt_aa!r}")
    return _ordered_amino_acids(allowed)


def _ordered_amino_acids(values: Iterable[str]) -> list[str]:
    allowed = set(values)
    return [aa for aa in PROTEINMPNN_ALPHABET if aa in allowed]


def _profile_residue_counts(
    *,
    profile_id: str,
    conservation_rows: list[dict[str, Any]],
    alignment_path: Path,
) -> dict[int, Counter[str]]:
    if not alignment_path.exists():
        raise FileNotFoundError(alignment_path)
    records = load_fasta_records(alignment_path, alphabet="protein", allow_gaps=True)
    source_sequences = [sequence for record_id, sequence in records.items() if record_id != TARGET_ALIGNMENT_ROW_ID]
    if not source_sequences:
        raise ValueError(f"{alignment_path} has no source rows after excluding {TARGET_ALIGNMENT_ROW_ID}")
    counts_by_position: dict[int, Counter[str]] = {}
    for row in conservation_rows:
        if str(row["profile_id"]) != profile_id:
            continue
        position = int(row["canonical_position"])
        column_index = int(row["msa_column"]) - 1
        counts_by_position[position] = Counter(
            sequence[column_index].upper()
            for sequence in source_sequences
            if sequence[column_index] != "-" and sequence[column_index].upper() in STANDARD_AMINO_ACIDS
        )
    if not counts_by_position:
        raise ValueError(f"No conservation rows found for {profile_id}")
    return counts_by_position
