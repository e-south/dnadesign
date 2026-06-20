"""Test fixtures for Eco1 conservation source-sequence bundles."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.aligner.msa import write_fasta_records

TARGET_ROW_ID = "eco1_rt_ec86kit_reference"
PROFILE_IDS = ("broad_retron_rt", "eco1_like_retron_rt")


def target_sequence(output_root: Path) -> str:
    rows = pq.read_table(output_root / "residue_map.parquet").to_pylist()
    return "".join(str(row["wt_aa"]) for row in rows)


def write_source_cache(
    tmp_path: Path,
    target: str,
    *,
    omit_accession: str | None = None,
    provider_override: str | None = None,
    record_id_override: str | None = None,
    omit_exclusion_reason: bool = False,
) -> Path:
    cache_root = tmp_path / "source_cache"
    provider_root = cache_root / "provider_caches"
    provider_root.mkdir(parents=True)

    ncbi_records = {
        accession: sequence
        for accession, sequence in {
            "WP_BROAD_1": target,
            "WP_ECO1_1": mutate(target, 12, "V"),
        }.items()
        if accession != omit_accession
    }
    write_fasta_records(provider_root / "ncbi_protein_efetch.fasta", ncbi_records)
    write_fasta_records(
        provider_root / "bv_brc_feature_protein_fasta.fasta",
        {
            "fig|BROAD.1": mutate(target, 8, "A"),
            "fig|ECO1.1": mutate(target, 16, "G"),
        },
    )

    excluded_record = {
        "profile_id": "broad_retron_rt",
        "record_id": "broad_missing",
        "provider_id": "ncbi_protein_efetch",
        "accession": "WP_MISSING",
        "status": "excluded",
    }
    if not omit_exclusion_reason:
        excluded_record["exclusion_reason"] = "provider_unresolved"

    records = [
        {
            "profile_id": "broad_retron_rt",
            "record_id": record_id_override or "broad_ncbi_1",
            "provider_id": provider_override or "ncbi_protein_efetch",
            "accession": "WP_BROAD_1",
            "status": "included",
        },
        {
            "profile_id": "broad_retron_rt",
            "record_id": "broad_bvbrc_1",
            "provider_id": "bv_brc_feature_protein_fasta",
            "accession": "fig|BROAD.1",
            "status": "included",
        },
        excluded_record,
        {
            "profile_id": "eco1_like_retron_rt",
            "record_id": "eco1_ncbi_1",
            "provider_id": "ncbi_protein_efetch",
            "accession": "WP_ECO1_1",
            "status": "included",
        },
        {
            "profile_id": "eco1_like_retron_rt",
            "record_id": "eco1_bvbrc_1",
            "provider_id": "bv_brc_feature_protein_fasta",
            "accession": "fig|ECO1.1",
            "status": "included",
        },
    ]
    _write_source_records(cache_root, records)
    return cache_root


def write_sufficient_source_cache(
    tmp_path: Path,
    target: str,
    *,
    records_per_profile: int = 20,
    placeholder_accession: bool = False,
) -> Path:
    cache_root = tmp_path / "source_cache"
    provider_root = cache_root / "provider_caches"
    provider_root.mkdir(parents=True)

    ncbi_records: dict[str, str] = {}
    bvbrc_records: dict[str, str] = {}
    source_rows: list[dict[str, str]] = []
    for profile_index, profile_id in enumerate(PROFILE_IDS, start=1):
        for record_index in range(records_per_profile):
            provider_id = "ncbi_protein_efetch" if record_index % 2 == 0 else "bv_brc_feature_protein_fasta"
            is_placeholder = placeholder_accession and profile_index == 1 and record_index == 0
            if provider_id == "ncbi_protein_efetch":
                accession = "WP_BROAD_1" if is_placeholder else f"WP_{100000000 + profile_index * 100 + record_index}.1"
                ncbi_records[accession] = mutate(target, ((record_index + profile_index) % 300) + 1, "A")
            else:
                accession = "fig|BROAD.1" if is_placeholder else f"fig|123456.{profile_index}.peg.{1000 + record_index}"
                bvbrc_records[accession] = mutate(target, ((record_index + profile_index) % 300) + 1, "G")
            source_rows.append(
                {
                    "profile_id": profile_id,
                    "record_id": f"{profile_id}_homolog_{record_index + 1:02d}",
                    "provider_id": provider_id,
                    "accession": accession,
                    "status": "included",
                }
            )
        source_rows.append(
            {
                "profile_id": profile_id,
                "record_id": f"{profile_id}_excluded_01",
                "provider_id": "ncbi_protein_efetch",
                "accession": f"WP_{200000000 + profile_index}.1",
                "status": "excluded",
                "exclusion_reason": "outside_identity_range",
            }
        )

    write_fasta_records(provider_root / "ncbi_protein_efetch.fasta", ncbi_records)
    write_fasta_records(provider_root / "bv_brc_feature_protein_fasta.fasta", bvbrc_records)
    _write_source_records(cache_root, source_rows)
    return cache_root


def mutate(sequence: str, position: int, aa: str) -> str:
    return sequence[: position - 1] + aa + sequence[position:]


def _write_source_records(cache_root: Path, records: list[dict[str, str]]) -> None:
    (cache_root / "source_records.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "study_id": "eco1_rt_repack",
                "records": records,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
