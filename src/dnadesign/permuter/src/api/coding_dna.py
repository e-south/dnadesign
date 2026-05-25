"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/api/coding_dna.py

Coding-DNA-backed DMS generation for the public API.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dnadesign.permuter.src.api.contracts import (
    CodingDnaDmsRequest,
    PermuterResult,
    VariantRecord,
    with_permuter_metadata,
)
from dnadesign.permuter.src.api.ids import request_id, stable_id
from dnadesign.permuter.src.protocols.base import assert_dna
from dnadesign.permuter.src.protocols.combine.codon_utils import (
    CodonTable,
    aa_to_best_codon,
    load_codon_table,
)

_CODON_LENGTH = 3
_PROTEIN_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")


def generate_coding_dna_dms(request: CodingDnaDmsRequest) -> PermuterResult:
    ref_name = _require_name(request.ref_name)
    sequence = str(request.sequence)
    assert_dna(sequence)
    if len(sequence) % _CODON_LENGTH != 0:
        raise ValueError("Coding DNA sequence length must be divisible by 3")
    if request.codon_policy != "top":
        raise ValueError(f"Unsupported codon_policy: {request.codon_policy!r}")
    codon_table_path = _require_codon_table(request.codon_table)
    table = load_codon_table(codon_table_path)
    positions = _normalize_positions(len(sequence) // _CODON_LENGTH, request.positions)
    alternates = _normalize_alternates(request.alternate_amino_acids, available=table.aa2codons.keys())
    sequence_upper = sequence.upper()
    expected_count = _expected_variant_count(
        sequence_upper=sequence_upper,
        positions=positions,
        alternates=alternates,
        table=table,
    )
    _enforce_max_variants(expected_count=expected_count, max_variants=request.max_variants)

    records: list[VariantRecord] = []
    for pos1 in positions:
        codon_start = (pos1 - 1) * _CODON_LENGTH
        wt_codon = sequence_upper[codon_start : codon_start + _CODON_LENGTH]
        wt_aa = table.codon2aa.get(wt_codon)
        if wt_aa is None:
            raise ValueError(f"Reference codon {wt_codon!r} at AA position {pos1} is absent from the codon table")
        for alt_aa in alternates:
            if alt_aa == wt_aa:
                continue
            new_codon = aa_to_best_codon(table, alt_aa)
            variant_sequence, nt_changes = _replace_codon_preserving_case(sequence, codon_start, new_codon)
            records.append(
                VariantRecord(
                    id=stable_id("coding-dna", ref_name, pos1, wt_aa, alt_aa, variant_sequence),
                    ref_name=ref_name,
                    bio_type="dna",
                    sequence=variant_sequence,
                    modifications=(
                        f"codon i={pos1 - 1} wt={wt_codon} new={new_codon} aa={alt_aa}",
                        f"aa pos={pos1} wt={wt_aa} alt={alt_aa}",
                        *(f"nt pos={nt_pos} wt={wt} alt={alt}" for nt_pos, wt, alt in nt_changes),
                    ),
                    metadata=with_permuter_metadata(
                        request.metadata,
                        {
                            "protocol": "coding_dna_dms",
                            "aa_pos": pos1,
                            "aa_wt": wt_aa,
                            "aa_alt": alt_aa,
                            "codon_index": pos1 - 1,
                            "codon_wt": wt_codon,
                            "codon_new": new_codon,
                            "codon_policy": request.codon_policy,
                            "codon_table": str(codon_table_path),
                        },
                    ),
                )
            )

    if not records:
        raise ValueError("Coding DNA DMS request produced zero variants")
    return PermuterResult(
        request_id=request_id(
            "coding-dna-dms",
            ref_name,
            sequence_upper,
            (*positions, *alternates, request.codon_policy, codon_table_path),
        ),
        ref_name=ref_name,
        bio_type="dna",
        reference_sequence=sequence,
        records=tuple(records),
        metadata=with_permuter_metadata(
            request.metadata,
            {
                "protocol": "coding_dna_dms",
                "codon_policy": request.codon_policy,
                "codon_table": str(codon_table_path),
                "positions": positions,
                "alternate_amino_acids": alternates,
            },
        ),
    )


def _require_name(value: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise ValueError("ref_name is required")
    return name


def _require_codon_table(value: str | Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("codon_table is required for CodingDnaDmsRequest")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"codon_table not found: {path}")
    return path


def _normalize_positions(codon_count: int, raw: tuple[int, ...]) -> tuple[int, ...]:
    if not raw:
        return tuple(range(1, codon_count + 1))
    positions = tuple(int(pos) for pos in raw)
    bad = [pos for pos in positions if pos < 1 or pos > codon_count]
    if bad:
        raise ValueError(f"Coding DNA position(s) out of bounds for {codon_count} codons: {bad}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"Duplicate coding DNA positions are not allowed: {positions}")
    return positions


def _normalize_alternates(raw: tuple[str, ...], *, available: Iterable[str]) -> tuple[str, ...]:
    available_set = {str(aa).upper() for aa in available}
    if raw:
        alternates = tuple(str(aa).strip().upper() for aa in raw)
        bad = [aa for aa in alternates if aa not in _PROTEIN_ALPHABET]
        if bad:
            raise ValueError(f"Unsupported alternate amino acid(s): {bad}")
        missing = [aa for aa in alternates if aa not in available_set]
        if missing:
            raise ValueError(f"Alternate amino acid(s) absent from codon table: {missing}")
    else:
        alternates = tuple(aa for aa in _PROTEIN_ALPHABET if aa in available_set)
    if not alternates:
        raise ValueError("No alternate amino acids are available for CodingDnaDmsRequest")
    if len(set(alternates)) != len(alternates):
        raise ValueError(f"Duplicate alternate amino acids are not allowed: {alternates}")
    return alternates


def _expected_variant_count(
    *,
    sequence_upper: str,
    positions: tuple[int, ...],
    alternates: tuple[str, ...],
    table: CodonTable,
) -> int:
    count = 0
    for pos1 in positions:
        codon_start = (pos1 - 1) * _CODON_LENGTH
        wt_codon = sequence_upper[codon_start : codon_start + _CODON_LENGTH]
        wt_aa = table.codon2aa.get(wt_codon)
        if wt_aa is None:
            raise ValueError(f"Reference codon {wt_codon!r} at AA position {pos1} is absent from the codon table")
        count += sum(1 for alt_aa in alternates if alt_aa != wt_aa)
    return count


def _enforce_max_variants(*, expected_count: int, max_variants: int | None) -> None:
    if max_variants is None:
        return
    limit = int(max_variants)
    if limit < 1:
        raise ValueError("max_variants must be >= 1 when provided")
    if expected_count > limit:
        raise ValueError(f"Coding DNA DMS request would produce {expected_count} variants, above max_variants={limit}")


def _replace_codon_preserving_case(
    sequence: str, start_0: int, new_codon_upper: str
) -> tuple[str, tuple[tuple[int, str, str], ...]]:
    chars = list(sequence)
    nt_changes: list[tuple[int, str, str]] = []
    for offset, new_upper in enumerate(new_codon_upper):
        index = start_0 + offset
        old_upper = chars[index].upper()
        if old_upper != new_upper:
            nt_changes.append((index + 1, old_upper, new_upper))
        chars[index] = new_upper if chars[index].isupper() else new_upper.lower()
    return "".join(chars), tuple(nt_changes)
