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
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from .sequence_contract import MUTATION_RE

_DESIGN_GROUP_LABELS = {
    "distal_scaffold_repack": "distal scaffold repack",
    "peripheral_shell_repack": "peripheral nucleic-acid-facing shell repack",
    "combined_peripheral_and_distal_repack": "combined peripheral and distal repack",
}

_PROTECTED_REASON_LABELS = {
    "motif_context_naxxh": "Protected motif context: NAxxH",
    "motif_context_yadd": "Protected motif context: YADD",
    "motif_context_vtg": "Protected motif context: VTG",
    "direct_retained_dna_rna_contact_le5a": "Protected: direct DNA/RNA contacts (<=5 A)",
    "wang_thumb_contact_track": "Protected: Wang thumb-contact track",
    "c_terminal_thumb_context_255_311": "Protected: primer-recognition context (255-311)",
    "conserved_core_clade9_25pct_plurality": "Protected: conserved/core positions",
}


def build_genbank_record(
    *, sequence_id: str, dna: str, protein: str, metadata: dict[str, Any], policy_rows: list[dict[str, Any]]
) -> SeqRecord:
    """Build one full-CDS record with protected contexts and substitutions."""

    parsed_mutations = _parse_mutations(metadata["mutation_tokens"])
    mutation_tokens = [mutation[0] for mutation in parsed_mutations]
    mutation_count = len(parsed_mutations)
    design_group = _DESIGN_GROUP_LABELS.get(str(metadata["design_group_id"]), str(metadata["design_group_id"]))
    record = SeqRecord(
        Seq(dna),
        id=sequence_id,
        name=sequence_id[:16],
        description=f"Eco1 reverse transcriptase variant with {mutation_count} amino-acid substitutions",
    )
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = "linear"
    record.annotations["source"] = "synthetic DNA construct"
    record.annotations["organism"] = "synthetic construct"
    record.annotations["taxonomy"] = ["other sequences", "artificial sequences"]
    record.annotations["comment"] = (
        f"AMINO-ACID SUBSTITUTIONS ({mutation_count}): {', '.join(mutation_tokens)}. "
        "Notation is reference residue, Eco1 RT residue number, then designed residue."
    )
    record.annotations["selection_rank"] = metadata["selection_rank"]
    record.annotations["design_group_id"] = metadata["design_group_id"]
    record.annotations["wang_alpha1_r13_review_status"] = metadata["wang_alpha1_r13_review_status"]
    record.annotations["wang_alpha1_mutation_count"] = metadata["wang_alpha1_mutation_count"]
    record.annotations["policy_id"] = metadata["policy_id"]
    record.annotations["candidate_id"] = metadata["candidate_id"]
    record.features.append(
        SeqFeature(
            FeatureLocation(0, len(dna), strand=1),
            type="gene",
            qualifiers={
                "label": [sequence_id],
                "gene": ["eco1_rt"],
                "note": [f"Designed Eco1 reverse transcriptase variant {sequence_id}."],
            },
        )
    )
    record.features.append(
        SeqFeature(
            FeatureLocation(0, len(dna), strand=1),
            type="CDS",
            qualifiers={
                "label": [f"{sequence_id} Eco1 RT CDS"],
                "gene": ["eco1_rt"],
                "product": ["Eco1 reverse transcriptase variant"],
                "codon_start": ["1"],
                "transl_table": ["11"],
                "translation": [protein],
                "note": [
                    f"Designed under the {design_group} policy. Within-group selection order: "
                    f"{metadata['within_group_rank']}; panel order: {metadata['selection_rank']}. "
                    f"F10 state: {metadata['wang_alpha1_f10_substitution']}; "
                    f"R13 state: {metadata['wang_alpha1_r13_substitution']}. "
                    "The RT-msDNA assembly state was not established. "
                    f"Source candidate: {metadata['candidate_id']}."
                ],
            },
        )
    )
    record.features.append(
        SeqFeature(
            FeatureLocation(0, len(dna), strand=1),
            type="misc_feature",
            qualifiers={
                "label": [f"{mutation_count} amino-acid substitutions"],
                "note": [
                    "Compact notation: reference amino acid + Eco1 RT residue number + designed amino acid.",
                    f"Changes: {', '.join(mutation_tokens)}",
                ],
            },
        )
    )
    for token, reference, position, designed in parsed_mutations:
        record.features.append(
            SeqFeature(
                FeatureLocation((position - 1) * 3, position * 3, strand=1),
                type="variation",
                qualifiers={
                    "label": [token],
                    "standard_name": [token],
                    "note": [f"{token}: reference {reference} at Eco1 RT residue {position} changed to {designed}."],
                },
            )
        )
    record.features.extend(_wang_alpha1_review_features(metadata))
    record.features.extend(_protected_context_features(policy_rows, str(metadata["policy_id"])))
    return record


def _parse_mutations(raw_tokens: Any) -> list[tuple[str, str, int, str]]:
    if not isinstance(raw_tokens, list):
        raise ValueError("GenBank mutation tokens must be a list")
    parsed: list[tuple[str, str, int, str]] = []
    for token in raw_tokens:
        if not isinstance(token, str):
            raise ValueError(f"invalid mutation token in GenBank export: {token!r}")
        match = MUTATION_RE.fullmatch(token)
        if match is None:
            raise ValueError(f"invalid mutation token in GenBank export: {token!r}")
        reference, position_text, designed = match.groups()
        parsed.append((token, reference, int(position_text), designed))
    return parsed


def _wang_alpha1_review_features(metadata: dict[str, Any]) -> list[SeqFeature]:
    f10_substitution = str(metadata["wang_alpha1_f10_substitution"])
    r13_substitution = str(metadata["wang_alpha1_r13_substitution"])
    r13a_evidence_match = bool(metadata["wang_r13a_interface_disruption_evidence_match"])
    assembly_status = str(metadata["rt_msdna_oligomeric_state_review_status"])
    return [
        SeqFeature(
            FeatureLocation((4 - 1) * 3, 16 * 3, strand=1),
            type="misc_feature",
            qualifiers={
                "label": ["Review: Wang alpha-1 interface (4-16)"],
                "note": [
                    "Wang et al. 2022 alpha-1 protomer-interface context at Eco1 residues 4-16. "
                    f"This sequence has {metadata['wang_alpha1_mutation_count']} substitutions in that context. "
                    f"F10 state: {f10_substitution}; R13 state: {r13_substitution}."
                ],
            },
        ),
        SeqFeature(
            FeatureLocation((13 - 1) * 3, 13 * 3, strand=1),
            type="misc_feature",
            qualifiers={
                "label": ["Review: Wang R13 interface residue"],
                "note": [
                    f"R13 state: {r13_substitution}. Matches the tested R13A substitution: "
                    f"{'yes' if r13a_evidence_match else 'no'}. RT-msDNA assembly state: {assembly_status}. "
                    "Wang et al. 2022 reported that R13A disrupted the two-protomer contact while retaining "
                    "msDNA and the tested antiphage activity."
                ],
            },
        ),
    ]


def _protected_context_features(rows: list[dict[str, Any]], policy_id: str) -> list[SeqFeature]:
    by_reason: dict[str, set[int]] = {}
    policy_positions = [row for row in rows if str(row.get("policy_id")) == policy_id]
    if len(policy_positions) != 320:
        raise ValueError(f"generation policy {policy_id!r} must contain exactly 320 position rows")
    for row in policy_positions:
        position = int(row["eco1_position"])
        for reason in row.get("protected_reason_codes") or []:
            by_reason.setdefault(str(reason), set()).add(position)
    unknown_reasons = sorted(set(by_reason) - set(_PROTECTED_REASON_LABELS))
    if unknown_reasons:
        raise ValueError(f"generation policy {policy_id!r} has unknown protected reason codes: {unknown_reasons}")
    if not by_reason:
        raise ValueError(f"generation policy {policy_id!r} has no protected-context annotations")
    features: list[SeqFeature] = []
    for reason, label in _PROTECTED_REASON_LABELS.items():
        positions = by_reason.get(reason)
        if not positions:
            raise ValueError(f"generation policy {policy_id!r} is missing protected reason code {reason!r}")
        ranges = _contiguous_ranges(positions)
        parts = [FeatureLocation((start - 1) * 3, end * 3, strand=1) for start, end in ranges]
        location = parts[0] if len(parts) == 1 else CompoundLocation(parts, operator="join")
        features.append(
            SeqFeature(
                location,
                type="misc_feature",
                qualifiers={
                    "label": [label],
                    "note": [f"{label}. Eco1 RT residues {_format_ranges(ranges)}. Policy reason code: {reason}."],
                },
            )
        )
    return features


def _contiguous_ranges(positions: set[int]) -> list[tuple[int, int]]:
    ordered = sorted(positions)
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position != previous + 1:
            ranges.append((start, previous))
            start = position
        previous = position
    ranges.append((start, previous))
    return ranges


def _format_ranges(ranges: list[tuple[int, int]]) -> str:
    return ", ".join(str(start) if start == end else f"{start}-{end}" for start, end in ranges)


__all__ = ["build_genbank_record"]
