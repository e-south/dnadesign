"""Deterministic codon design and sequence QC for Eco1 RT fragments."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from .sequence_contract import FORBIDDEN_SITES


def highest_frequency_codons(path: Path) -> dict[str, str]:
    """Return one highest-frequency E. coli codon per canonical amino acid."""

    best: dict[str, tuple[float, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            amino_acid = str(row["amino_acid"])
            if amino_acid == "*":
                continue
            try:
                frequency = float(row["frequency"])
            except ValueError as error:
                raise ValueError(f"invalid E. coli codon frequency for {row['codon']}") from error
            choice = (frequency, str(row["codon"]).upper())
            if amino_acid not in best or choice > best[amino_acid]:
                best[amino_acid] = choice
    if set(best) != set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("E. coli codon table does not cover all canonical amino acids")
    return {amino_acid: choice[1] for amino_acid, choice in best.items()}


def encode_full_cds(wt_dna: str, wt: str, candidate: str, codons: dict[str, str]) -> str:
    """Preserve WT codons at unchanged residues and recode substitutions."""

    encoded = [
        wt_dna[index * 3 : index * 3 + 3] if ref == alt else codons[alt]
        for index, (ref, alt) in enumerate(zip(wt, candidate, strict=True))
    ]
    encoded.append(wt_dna[-3:])
    return "".join(encoded)


def sequence_qc(dna: str) -> dict[str, int | float]:
    """Return deterministic fragment-level manufacturability checks."""

    counts = Counter(dna[index : index + 20] for index in range(len(dna) - 19))
    return {
        "gc_fraction": round((dna.count("G") + dna.count("C")) / len(dna), 6),
        "max_homopolymer_run": max(len(match.group(0)) for match in re.finditer(r"([ACGT])\1*", dna)),
        "repeated_20mer_count": sum(1 for count in counts.values() if count > 1),
        "forbidden_site_count": sum(dna.count(site) for site in FORBIDDEN_SITES),
    }


__all__ = ["encode_full_cds", "highest_frequency_codons", "sequence_qc"]
