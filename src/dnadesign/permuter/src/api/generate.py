"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/api/generate.py

Filesystem-free public variant generation.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from dnadesign.permuter.src.api.coding_dna import generate_coding_dna_dms
from dnadesign.permuter.src.api.contracts import (
    CodingDnaDmsRequest,
    NucleotideDmsRequest,
    PermuterResult,
    ProteinDmsRequest,
    VariantRecord,
    with_permuter_metadata,
)
from dnadesign.permuter.src.api.ids import request_id, stable_id
from dnadesign.permuter.src.protocols.base import assert_dna
from dnadesign.permuter.src.protocols.dms.scan_dna import ScanDNA

_PROTEIN_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
_PROTEIN_RE = re.compile(r"^[A-Za-z*]+$")


def generate_variants(request: NucleotideDmsRequest | CodingDnaDmsRequest | ProteinDmsRequest) -> PermuterResult:
    if isinstance(request, NucleotideDmsRequest):
        return _generate_nucleotide_dms(request)
    if isinstance(request, CodingDnaDmsRequest):
        return generate_coding_dna_dms(request)
    if isinstance(request, ProteinDmsRequest):
        return _generate_protein_dms(request)
    raise TypeError(f"Unsupported Permuter request type: {type(request).__name__}")


def _generate_nucleotide_dms(request: NucleotideDmsRequest) -> PermuterResult:
    ref_name = _require_name(request.ref_name)
    sequence = str(request.sequence)
    assert_dna(sequence)
    regions = _normalize_dna_regions(len(sequence), request.regions)
    protocol = ScanDNA()
    rows = protocol.generate(
        ref_entry={"ref_name": ref_name, "sequence": sequence},
        params={"regions": [list(r) for r in regions]},
    )
    records = tuple(
        VariantRecord(
            id=stable_id("dna", row["sequence"]),
            ref_name=ref_name,
            bio_type="dna",
            sequence=str(row["sequence"]),
            modifications=tuple(str(token) for token in row["modifications"]),
            metadata=with_permuter_metadata(
                request.metadata,
                {
                    "protocol": "nucleotide_dms",
                    "nt_pos": int(row["nt_pos"]),
                    "nt_wt": str(row["nt_wt"]),
                    "nt_alt": str(row["nt_alt"]),
                },
            ),
        )
        for row in rows
    )
    return PermuterResult(
        request_id=request_id("dna-dms", ref_name, sequence, regions),
        ref_name=ref_name,
        bio_type="dna",
        reference_sequence=sequence,
        records=records,
        metadata=with_permuter_metadata(
            request.metadata,
            {
                "protocol": "nucleotide_dms",
                "regions": regions,
            },
        ),
    )


def _generate_protein_dms(request: ProteinDmsRequest) -> PermuterResult:
    ref_name = _require_name(request.ref_name)
    sequence = str(request.sequence).upper()
    _assert_protein(sequence)
    positions = _normalize_protein_positions(len(sequence), request.positions)
    records: list[VariantRecord] = []
    for pos1 in positions:
        wt = sequence[pos1 - 1]
        for alt in _PROTEIN_ALPHABET:
            if alt == wt:
                continue
            variant = sequence[: pos1 - 1] + alt + sequence[pos1:]
            records.append(
                VariantRecord(
                    id=stable_id("protein", variant),
                    ref_name=ref_name,
                    bio_type="protein",
                    sequence=variant,
                    modifications=(f"aa pos={pos1} wt={wt} alt={alt}",),
                    metadata=with_permuter_metadata(
                        request.metadata,
                        {
                            "protocol": "protein_dms",
                            "aa_pos": pos1,
                            "aa_wt": wt,
                            "aa_alt": alt,
                        },
                    ),
                )
            )
    return PermuterResult(
        request_id=request_id("protein-dms", ref_name, sequence, positions),
        ref_name=ref_name,
        bio_type="protein",
        reference_sequence=sequence,
        records=tuple(records),
        metadata=with_permuter_metadata(
            request.metadata,
            {
                "protocol": "protein_dms",
                "positions": positions,
            },
        ),
    )


def _require_name(value: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise ValueError("ref_name is required")
    return name


def _normalize_dna_regions(seq_len: int, raw: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not raw:
        return ((0, seq_len),)
    regions: list[tuple[int, int]] = []
    for region in raw:
        if len(region) != 2:
            raise ValueError(f"DNA region must be a [start, end) pair; got {region!r}")
        start, end = int(region[0]), int(region[1])
        if not (0 <= start < end <= seq_len):
            raise ValueError(f"DNA region out of bounds for length {seq_len}: {region!r}")
        regions.append((start, end))
    return tuple(regions)


def _assert_protein(sequence: str) -> None:
    if not sequence or not _PROTEIN_RE.fullmatch(sequence):
        raise ValueError("Sequence must be protein letters")
    invalid = sorted(set(sequence) - set(_PROTEIN_ALPHABET) - {"*"})
    if invalid:
        raise ValueError(f"Sequence contains unsupported protein residue(s): {invalid}")


def _normalize_protein_positions(seq_len: int, raw: tuple[int, ...]) -> tuple[int, ...]:
    if not raw:
        return tuple(range(1, seq_len + 1))
    positions = tuple(int(pos) for pos in raw)
    bad = [pos for pos in positions if pos < 1 or pos > seq_len]
    if bad:
        raise ValueError(f"Protein position(s) out of bounds for length {seq_len}: {bad}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"Duplicate protein positions are not allowed: {positions}")
    return positions
