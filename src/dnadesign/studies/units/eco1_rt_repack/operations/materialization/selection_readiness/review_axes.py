"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/review_axes.py

Interpretable review axes for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from ast import literal_eval
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dnadesign.aligner.msa import load_fasta_records
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)

CLADE9_PROFILE_ID = "ec86_clade9_conservation_v1"
SUBTYPE_PROFILE_ID = "ec86_iia3_cluster42_1_conservation_v1"
TARGET_ROW_ID = "eco1_rt_ec86kit_reference"
RARE_RESIDUE_FREQUENCY = 0.01
NA_FACING_DISTANCE_ANGSTROM = 10.0
DIRECT_CONTACT_DISTANCE_ANGSTROM = 5.0
WANG_THUMB_CONTACT_TRACK_POSITIONS = {238, 239, 240, 249, 257, 261, 264, 298}
C_TERMINAL_PRIMER_RNA_RECOGNITION_POSITIONS = frozenset(range(255, 312))
_MUTATION_RE = re.compile(r"([A-Z])(\d+)([A-Z*])")
_BASIC = set("KRH")
_ACIDIC = set("DE")
_PROLINE_GLYCINE = set("PG")


@dataclass(frozen=True)
class Mutation:
    """One canonical Eco1 amino-acid substitution."""

    wt_aa: str
    position: int
    alt_aa: str


@dataclass(frozen=True)
class ProfileSupport:
    """Per-position MSA support for alternate residues."""

    non_gap_count: int
    residue_counts: Counter[str]


def build_review_axis_by_candidate(
    *,
    candidate_rows: Sequence[dict[str, object]],
    conservation_profile_rows: Sequence[dict[str, object]],
    clade9_alignment_path: Path,
    subtype_alignment_path: Path,
    contact_geometry_rows: Sequence[dict[str, object]],
    mask_residues: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Build natural-support, mutation-geography, and local-chemistry fields."""

    profile_support = {
        CLADE9_PROFILE_ID: _profile_support(
            profile_id=CLADE9_PROFILE_ID,
            conservation_profile_rows=conservation_profile_rows,
            alignment_path=clade9_alignment_path,
        ),
        SUBTYPE_PROFILE_ID: _profile_support(
            profile_id=SUBTYPE_PROFILE_ID,
            conservation_profile_rows=conservation_profile_rows,
            alignment_path=subtype_alignment_path,
        ),
    }
    contact_by_position = {int(row["canonical_position"]): row for row in contact_geometry_rows}
    mask_by_position = {int(row["canonical_position"]): row for row in mask_residues}
    profile_by_class = {spec.design_class_id: spec.conservation_profile_id for spec in ALL_SPECS}

    axes: dict[str, dict[str, object]] = {}
    for candidate in candidate_rows:
        if str(candidate.get("status")) != "accepted":
            continue
        candidate_id = str(candidate["candidate_id"])
        mutations = _parse_mutations(candidate.get("canonical_mutations"), candidate_id=candidate_id)
        class_id = str(candidate["design_class_id"])
        if class_id not in profile_by_class:
            raise ValueError(f"Unknown Eco1 design class id for review-axis selection: {class_id}")
        selected_profile_id = profile_by_class[class_id]
        clade9 = _natural_support(mutations, profile_support[CLADE9_PROFILE_ID])
        subtype = _natural_support(mutations, profile_support[SUBTYPE_PROFILE_ID])
        selected = clade9 if selected_profile_id == CLADE9_PROFILE_ID else subtype
        axes[candidate_id] = {
            **_prefix("clade9", clade9),
            **_prefix("subtype", subtype),
            "selection_support_profile_id": selected_profile_id,
            "selection_support_alt_observed_fraction": selected["alt_observed_fraction"],
            "selection_support_alt_frequency_mean": selected["alt_frequency_mean"],
            "selection_support_unobserved_mutation_count": selected["unobserved_mutation_count"],
            **_mutation_geography(
                mutations,
                contact_by_position=contact_by_position,
                mask_by_position=mask_by_position,
            ),
        }
    return axes


def _profile_support(
    *,
    profile_id: str,
    conservation_profile_rows: Sequence[dict[str, object]],
    alignment_path: Path,
) -> dict[int, ProfileSupport]:
    records = load_fasta_records(alignment_path, alphabet="protein", allow_gaps=True)
    source_sequences = [sequence for record_id, sequence in records.items() if record_id != TARGET_ROW_ID]
    if not source_sequences:
        raise ValueError(f"{alignment_path} has no source rows after excluding {TARGET_ROW_ID}")
    support: dict[int, ProfileSupport] = {}
    for row in conservation_profile_rows:
        if str(row["profile_id"]) != profile_id:
            continue
        position = int(row["canonical_position"])
        column_index = int(row["msa_column"]) - 1
        counts = Counter(sequence[column_index] for sequence in source_sequences if sequence[column_index] != "-")
        support[position] = ProfileSupport(non_gap_count=sum(counts.values()), residue_counts=counts)
    if not support:
        raise ValueError(f"No conservation rows found for {profile_id}")
    return support


def _natural_support(mutations: Sequence[Mutation], support: dict[int, ProfileSupport]) -> dict[str, object]:
    observed = 0
    unobserved = 0
    rare_or_unobserved = 0
    frequencies: list[float] = []
    for mutation in mutations:
        row = support.get(mutation.position)
        if row is None or row.non_gap_count == 0:
            continue
        count = row.residue_counts.get(mutation.alt_aa, 0)
        frequency = count / row.non_gap_count
        frequencies.append(frequency)
        observed += int(count > 0)
        unobserved += int(count == 0)
        rare_or_unobserved += int(frequency < RARE_RESIDUE_FREQUENCY)
    total = len(frequencies)
    return {
        "designed_mutation_count": len(mutations),
        "supportable_mutation_count": total,
        "alt_observed_fraction": observed / total if total else None,
        "alt_frequency_mean": sum(frequencies) / total if total else None,
        "unobserved_mutation_count": unobserved,
        "rare_or_unobserved_mutation_count": rare_or_unobserved,
    }


def _mutation_geography(
    mutations: Sequence[Mutation],
    *,
    contact_by_position: dict[int, dict[str, object]],
    mask_by_position: dict[int, dict[str, object]],
) -> dict[str, object]:
    catalytic_or_direct = 0
    na_facing = 0
    thumb_track = 0
    c_terminal_primer_rna_recognition = 0
    distal = 0
    charge_delta = 0
    basic_gain = 0
    basic_loss = 0
    acidic_gain = 0
    proline_glycine_gain = 0
    for mutation in mutations:
        position = mutation.position
        mask = mask_by_position.get(position, {})
        contact = contact_by_position.get(position, {})
        distance = _float_or_none(contact.get("nearest_context_atom_distance_angstrom"))
        is_catalytic_or_direct = (
            bool(mask.get("motif_protected"))
            or bool(mask.get("wang_ec86_direct_contact_prior"))
            or bool(mask.get("direct_retained_dna_rna_contact_5a"))
            or (distance is not None and distance <= DIRECT_CONTACT_DISTANCE_ANGSTROM)
        )
        is_thumb_track = position in WANG_THUMB_CONTACT_TRACK_POSITIONS
        is_c_terminal_primer_rna_recognition = position in C_TERMINAL_PRIMER_RNA_RECOGNITION_POSITIONS
        is_na_facing = (distance is not None and distance <= NA_FACING_DISTANCE_ANGSTROM) or is_thumb_track
        catalytic_or_direct += int(is_catalytic_or_direct)
        thumb_track += int(is_thumb_track)
        c_terminal_primer_rna_recognition += int(is_c_terminal_primer_rna_recognition)
        na_facing += int(is_na_facing and not is_catalytic_or_direct)
        distal += int(not is_catalytic_or_direct and not is_na_facing)
        if is_na_facing:
            charge_delta += _charge(mutation.alt_aa) - _charge(mutation.wt_aa)
            basic_gain += int(mutation.wt_aa not in _BASIC and mutation.alt_aa in _BASIC)
            basic_loss += int(mutation.wt_aa in _BASIC and mutation.alt_aa not in _BASIC)
            acidic_gain += int(mutation.wt_aa not in _ACIDIC and mutation.alt_aa in _ACIDIC)
            proline_glycine_gain += int(mutation.wt_aa not in _PROLINE_GLYCINE and mutation.alt_aa in _PROLINE_GLYCINE)
    warnings = basic_loss + acidic_gain + proline_glycine_gain
    chemistry_compatible = charge_delta >= 0 and acidic_gain <= basic_gain
    return {
        "catalytic_or_direct_contact_mutation_count": catalytic_or_direct,
        "nucleic_acid_facing_mutation_count": na_facing,
        "thumb_contact_track_mutation_count": thumb_track,
        "c_terminal_primer_rna_recognition_mutation_count": c_terminal_primer_rna_recognition,
        "distal_scaffold_mutation_count": distal,
        "nucleic_acid_facing_charge_delta": charge_delta,
        "nucleic_acid_facing_basic_gain_count": basic_gain,
        "nucleic_acid_facing_basic_loss_count": basic_loss,
        "nucleic_acid_facing_acidic_gain_count": acidic_gain,
        "nucleic_acid_facing_proline_glycine_gain_count": proline_glycine_gain,
        "nucleic_acid_facing_chemistry_warning_count": warnings,
        "nucleic_acid_facing_chemistry_compatible": chemistry_compatible,
        "nucleic_acid_facing_chemistry_gate_status": "passed" if chemistry_compatible else "incompatible",
    }


def _parse_mutations(value: object, *, candidate_id: str = "unknown") -> list[Mutation]:
    if value is None:
        return []
    values = _mutation_tokens(value, candidate_id=candidate_id)
    mutations: list[Mutation] = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, tuple):
            wt_aa, position, alt_aa = item
        else:
            match = _MUTATION_RE.fullmatch(str(item).strip())
            if match is None:
                raise ValueError(
                    f"Malformed canonical mutation for {candidate_id} at token {index}: {str(item).strip()!r}"
                )
            wt_aa, position, alt_aa = match.groups()
        mutations.append(Mutation(wt_aa=str(wt_aa), position=int(position), alt_aa=str(alt_aa)))
    return mutations


def _mutation_tokens(value: object, *, candidate_id: str) -> list[object]:
    if isinstance(value, list | tuple):
        return list(value)
    if hasattr(value, "tolist"):
        loaded = value.tolist()
        return list(loaded) if isinstance(loaded, list | tuple) else [loaded]
    text = str(value).strip()
    if not text:
        return []
    if _MUTATION_RE.fullmatch(text):
        return [text]
    try:
        loaded = literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Malformed canonical mutation list for {candidate_id}: {text!r}") from exc
    if isinstance(loaded, list | tuple):
        return list(loaded)
    if _MUTATION_RE.fullmatch(str(loaded).strip()):
        return [loaded]
    raise ValueError(f"Malformed canonical mutation list for {candidate_id}: {text!r}")


def _prefix(prefix: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _charge(amino_acid: str) -> int:
    if amino_acid in _BASIC:
        return 1
    if amino_acid in _ACIDIC:
        return -1
    return 0


def _float_or_none(value: object) -> float | None:
    return None if value is None else float(value)
