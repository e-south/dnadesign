"""Fixture helpers for Eco1 conservation roster-cache tests."""

from __future__ import annotations

import csv
from pathlib import Path


def write_roster_table(tmp_path: Path, *, extra_rows: list[dict[str, str]] | None = None) -> Path:
    rows = [roster_row("public_eco1", "WP_099010551.1", subtype="II-A3")]
    rows.extend(roster_row(f"broad_ncbi_{index}", f"WP_{100000000 + index}.1") for index in range(1, 22))
    rows.extend(
        roster_row(
            f"eco1_like_{index}",
            f"fig|123456.{index}.peg.{2000 + index}",
            subtype="II-A3",
            cluster="42_1",
            clade="9",
        )
        for index in range(1, 21)
    )
    rows.extend(extra_rows or [])
    path = tmp_path / "mestre_s1_fixture.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Node",
                "Accesione",
                "Retron subtype",
                "Cluster/domain",
                "RT clade",
                "source_cache_status",
                "exclusion_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def roster_row(
    node: str,
    accession: str,
    *,
    subtype: str = "other",
    cluster: str = "other",
    clade: str = "1",
    status: str = "included",
    exclusion_reason: str = "",
) -> dict[str, str]:
    return {
        "Node": node,
        "Accesione": accession,
        "Retron subtype": subtype,
        "Cluster/domain": cluster,
        "RT clade": clade,
        "source_cache_status": status,
        "exclusion_reason": exclusion_reason,
    }


def write_provider_sources(tmp_path: Path, *, omit_accessions: set[str] | None = None) -> Path:
    omit = omit_accessions or set()
    root = tmp_path / "provider_sources"
    root.mkdir()
    ncbi_records: dict[str, str] = {}
    bvbrc_records: dict[str, str] = {}
    for index in range(1, 22):
        accession = f"WP_{100000000 + index}.1"
        if accession not in omit:
            ncbi_records[accession] = protein_sequence(index)
    for index in range(1, 21):
        accession = f"fig|123456.{index}.peg.{2000 + index}"
        if accession not in omit:
            bvbrc_records[accession] = protein_sequence(index + 30)
    write_fasta(root / "ncbi_protein_efetch.fasta", ncbi_records)
    write_fasta(root / "bv_brc_feature_protein_fasta.fasta", bvbrc_records)
    return root


def write_fasta(path: Path, records: dict[str, str]) -> None:
    lines: list[str] = []
    for record_id, sequence in records.items():
        lines.extend([f">{record_id}", sequence])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def protein_sequence(offset: int) -> str:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(amino_acids[(index + offset) % len(amino_acids)] for index in range(320))
