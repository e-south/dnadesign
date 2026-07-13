"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/twist_handoff/genbank_export.py

Annotated GenBank records for the Eco1 RT synthesis handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from .sequence_contract import MUTATION_RE

_CONTEXT_COLUMNS = {
    "is_direct_contact_le_5a": "direct_contact_le_5a",
    "is_near_region_gt5_le10a": "near_retained_dna_rna_gt5_le10a",
    "is_wang_thumb_track": "wang_thumb_track",
    "is_c_terminal_thumb_context": "c_terminal_thumb_context",
    "is_conserved_core": "conserved_core",
}


def build_genbank_record(
    *, sequence_id: str, dna: str, protein: str, metadata: dict[str, Any], policy_rows: list[dict[str, Any]]
) -> SeqRecord:
    """Build one full-CDS record with protected contexts and substitutions."""

    record = SeqRecord(Seq(dna), id=sequence_id, name=sequence_id[:16], description="Eco1 RT Twist full-CDS handoff")
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = "linear"
    record.annotations["selection_rank"] = metadata["selection_rank"]
    record.annotations["design_group_id"] = metadata["design_group_id"]
    record.annotations["wang_alpha1_r13_review_status"] = metadata["wang_alpha1_r13_review_status"]
    record.annotations["wang_alpha1_mutation_count"] = metadata["wang_alpha1_mutation_count"]
    record.annotations["policy_id"] = metadata["policy_id"]
    record.annotations["candidate_id"] = metadata["candidate_id"]
    record.features.append(
        SeqFeature(
            FeatureLocation(0, len(dna), strand=1),
            type="CDS",
            qualifiers={
                "label": ["Eco1 RT full CDS"],
                "codon_start": ["1"],
                "transl_table": ["11"],
                "translation": [protein],
                "note": [
                    f"selection_rank={metadata['selection_rank']}; design_group_id={metadata['design_group_id']}; "
                    f"within_group_rank={metadata['within_group_rank']}; selection_slot={metadata['selection_slot']}; "
                    f"wang_alpha1_r13={metadata['wang_alpha1_r13_review_status']}; "
                    f"wang_alpha1_mutation_count={metadata['wang_alpha1_mutation_count']}; "
                    f"policy_id={metadata['policy_id']}; "
                    f"candidate_id={metadata['candidate_id']}"
                ],
            },
        )
    )
    record.features.extend(_wang_alpha1_review_features(metadata))
    for label, start, end in _protected_context_ranges(policy_rows, str(metadata["policy_id"])):
        record.features.append(
            SeqFeature(
                FeatureLocation((start - 1) * 3, end * 3, strand=1),
                type="misc_feature",
                qualifiers={"label": [f"protected_context:{label}"], "note": [f"Eco1 residues {start}-{end}"]},
            )
        )
    for token in metadata["mutation_tokens"]:
        match = MUTATION_RE.fullmatch(token)
        if match is None:
            raise ValueError(f"invalid mutation token in GenBank export: {token!r}")
        position = int(match.group(2))
        record.features.append(
            SeqFeature(
                FeatureLocation((position - 1) * 3, position * 3, strand=1),
                type="variation",
                qualifiers={"label": [token], "note": [f"amino acid substitution {token}"]},
            )
        )
    return record


def _wang_alpha1_review_features(metadata: dict[str, Any]) -> list[SeqFeature]:
    r13_status = str(metadata["wang_alpha1_r13_review_status"])
    return [
        SeqFeature(
            FeatureLocation((4 - 1) * 3, 16 * 3, strand=1),
            type="misc_feature",
            qualifiers={
                "label": ["wang_alpha1_interface_review"],
                "note": [
                    "Eco1 residues 4-16; Wang et al. 2022 alpha-1 protomer-interface context; "
                    f"mutation count={metadata['wang_alpha1_mutation_count']}"
                ],
            },
        ),
        SeqFeature(
            FeatureLocation((13 - 1) * 3, 13 * 3, strand=1),
            type="misc_feature",
            qualifiers={
                "label": ["wang_alpha1_R13_review"],
                "note": [
                    f"observed_status={r13_status}; Wang et al. 2022 reported that R13A disrupted the "
                    "two-protomer contact while retaining msDNA production and antiphage defence"
                ],
            },
        ),
    ]


def _protected_context_ranges(rows: list[dict[str, Any]], policy_id: str) -> list[tuple[str, int, int]]:
    by_label: dict[str, set[int]] = {}
    policy_positions = [row for row in rows if str(row.get("policy_id")) == policy_id]
    if len(policy_positions) != 320:
        raise ValueError(f"generation policy {policy_id!r} must contain exactly 320 position rows")
    for row in policy_positions:
        position = int(row["eco1_position"])
        for reason in row.get("protected_reason_codes") or []:
            by_label.setdefault(str(reason), set()).add(position)
        for column, label in _CONTEXT_COLUMNS.items():
            if row.get(column) is True:
                by_label.setdefault(label, set()).add(position)
    ranges: list[tuple[str, int, int]] = []
    for label, positions in sorted(by_label.items()):
        ordered = sorted(positions)
        start = previous = ordered[0]
        for position in ordered[1:]:
            if position != previous + 1:
                ranges.append((label, start, previous))
                start = position
            previous = position
        ranges.append((label, start, previous))
    if not ranges:
        raise ValueError(f"generation policy {policy_id!r} has no protected-context annotations")
    return ranges


__all__ = ["build_genbank_record"]
