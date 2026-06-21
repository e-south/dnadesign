"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/source_sequences/roster_cache/_fixtures.py

Fixture helpers for Eco1 conservation roster-cache tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def ec86kit_target_sequence() -> str:
    """Load the selected Eco1/Ec86 reference sequence used by the source contract."""

    repo_root = Path(__file__).resolve().parents[9]
    manifest = (repo_root / "../ec86kit/out/ec86_prot1/runs/2025-10-09T17-47-31Z/manifest.json").resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for step in payload["steps"]:
        config = step.get("config")
        if isinstance(config, dict) and isinstance(config.get("sequence"), str):
            return config["sequence"]
    raise ValueError(f"ec86kit manifest has no reference sequence: {manifest}")


def write_roster_table(tmp_path: Path, *, extra_rows: list[dict[str, str]] | None = None) -> Path:
    rows = [roster_row("public_eco1", "WP_099010551.1", subtype="II-A3", cluster="42_1", clade="9")]
    rows.extend(roster_row(f"clade9_ncbi_{index}", f"WP_{100000000 + index}.1", clade="9") for index in range(1, 22))
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


def write_provider_sources(
    tmp_path: Path,
    *,
    omit_accessions: set[str] | None = None,
    target_sequence: str | None = None,
) -> Path:
    omit = omit_accessions or set()
    target = target_sequence or ec86kit_target_sequence()
    root = tmp_path / "provider_sources"
    root.mkdir()
    ncbi_records: dict[str, str] = {}
    bvbrc_records: dict[str, str] = {}
    for index in range(1, 22):
        accession = f"WP_{100000000 + index}.1"
        if accession not in omit:
            ncbi_records[accession] = homolog_sequence(target, index)
    for index in range(1, 21):
        accession = f"fig|123456.{index}.peg.{2000 + index}"
        if accession not in omit:
            bvbrc_records[accession] = homolog_sequence(target, index + 30)
    write_fasta(root / "ncbi_protein_efetch.fasta", ncbi_records)
    write_fasta(root / "bv_brc_feature_protein_fasta.fasta", bvbrc_records)
    return root


def write_fasta(path: Path, records: dict[str, str]) -> None:
    lines: list[str] = []
    for record_id, sequence in records.items():
        lines.extend([f">{record_id}", sequence])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def protein_sequence(offset: int) -> str:
    """Create a generic provider-source protein sequence for acquisition tests."""

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(amino_acids[(index + offset) % len(amino_acids)] for index in range(320))


def homolog_sequence(target: str, offset: int) -> str:
    """Create a target-like homolog with motif anchors preserved and <90% identity."""

    protected = _motif_positions(target)
    letters = list(target)
    changed = 0
    for step in range(len(letters)):
        if changed >= 48:
            break
        index = (offset + step) % len(letters)
        if index not in protected:
            current = letters[index]
            replacement = "A" if current != "A" else "C"
            if replacement != current:
                letters[index] = replacement
                changed += 1
    if changed < 48:
        raise AssertionError("fixture target sequence did not have enough mutable non-motif positions")
    return "".join(letters)


def fragment_sequence(target: str) -> str:
    return target[:220]


def no_catalytic_core_sequence(target: str) -> str:
    return target.replace("D", "E")


def _motif_positions(sequence: str) -> set[int]:
    protected: set[int] = set()
    for pattern in (r"YADD|[A-Z]{2}DD", r"NA..H", r"VTG"):
        for match in re.finditer(pattern, sequence):
            protected.update(range(match.start(), match.end()))
    return protected
