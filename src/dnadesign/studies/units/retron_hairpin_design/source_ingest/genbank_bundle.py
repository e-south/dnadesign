"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/genbank_bundle.py

Parse MSD-region GenBank source bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from Bio import SeqIO

from .genbank_utils import sha256_file, sha256_text, variant_id, variant_sort_key
from .models import (
    MsdRegionIngestError,
    MsdRegionSourceBundle,
    NormalizedMsdRegionRecord,
    SkippedMsdSourceRecord,
)
from .payload_binding import PayloadBindingCatalog
from .record_normalization import normalize_msd_region_record
from .variant_sources import source_inputs_from_manifest, variant_source_manifest


def parse_msd_region_genbank(
    path: str | Path,
    *,
    payload_catalog: PayloadBindingCatalog | None = None,
) -> MsdRegionSourceBundle:
    """Parse an MSD-region GenBank file into decomposed per-variant records."""

    source_path = Path(path).expanduser().resolve()
    records = list(SeqIO.parse(source_path, "genbank"))
    normalized: list[NormalizedMsdRegionRecord] = []
    skipped: list[SkippedMsdSourceRecord] = []
    seen: set[str] = set()
    for record in records:
        retron_variant_id = variant_id(record)
        if retron_variant_id is None:
            skipped.append(
                SkippedMsdSourceRecord(
                    record_id=record.id,
                    reason="unresolved_variant_id",
                    sequence_length_nt=len(record.seq),
                )
            )
            continue
        if retron_variant_id in seen:
            raise MsdRegionIngestError(f"Duplicate MSD-region variant id in {source_path}: {retron_variant_id}")
        seen.add(retron_variant_id)
        normalized.append(
            normalize_msd_region_record(record, variant_id=retron_variant_id, payload_catalog=payload_catalog)
        )
    normalized.sort(key=lambda item: variant_sort_key(item.variant_id))
    return MsdRegionSourceBundle(
        source_path=source_path.as_posix(),
        source_sha256=sha256_file(source_path),
        source_record_count=len(records),
        records=tuple(normalized),
        skipped_records=tuple(skipped),
        source_kind="bulk_migration_genbank",
    )


def parse_msd_region_genbank_dir(
    path: str | Path,
    *,
    payload_catalog: PayloadBindingCatalog | None = None,
) -> MsdRegionSourceBundle:
    """Parse a directory of one-variant MSD-region GenBank source files."""

    source_dir = Path(path).expanduser().resolve()
    if not source_dir.is_dir():
        raise MsdRegionIngestError(f"MSD-region source directory does not exist: {source_dir}")
    source_files = sorted(
        item for pattern in ("*.gb", "*.gbk", "*.genbank") for item in source_dir.glob(pattern) if item.is_file()
    )
    if not source_files:
        raise MsdRegionIngestError(f"MSD-region source directory has no GenBank files: {source_dir}")
    records_by_variant: dict[str, NormalizedMsdRegionRecord] = {}
    source_inputs: list[dict[str, object]] = []
    source_record_count = 0
    skipped_records: list[SkippedMsdSourceRecord] = []
    for source_file in source_files:
        bundle = parse_msd_region_genbank(source_file, payload_catalog=payload_catalog)
        source_record_count += bundle.source_record_count
        skipped_records.extend(bundle.skipped_records)
        if bundle.skipped_records:
            skipped = ", ".join(record.record_id for record in bundle.skipped_records)
            raise MsdRegionIngestError(f"{source_file} contains unresolved GenBank records: {skipped}")
        if len(bundle.records) != 1:
            raise MsdRegionIngestError(
                f"{source_file} must contain exactly one resolvable retron MSD record; found {len(bundle.records)}."
            )
        record = bundle.records[0]
        if record.variant_id in records_by_variant:
            raise MsdRegionIngestError(
                f"Duplicate MSD-region variant id across source directory {source_dir}: {record.variant_id}"
            )
        records_by_variant[record.variant_id] = record
        source_inputs.append(
            {
                "variant_id": record.variant_id,
                "display_id": record.display_id,
                "source_path": source_file.as_posix(),
                "source_sha256": bundle.source_sha256,
                "source_record_count": bundle.source_record_count,
                "source_role": "variant_genbank_source",
            }
        )
    source_hash = sha256_text("\n".join(f"{item['variant_id']}\t{item['source_sha256']}" for item in source_inputs))
    source_manifest = variant_source_manifest(source_dir)
    return MsdRegionSourceBundle(
        source_path=source_dir.as_posix(),
        source_sha256=source_hash,
        source_record_count=source_record_count,
        records=tuple(sorted(records_by_variant.values(), key=lambda item: variant_sort_key(item.variant_id))),
        skipped_records=tuple(skipped_records),
        source_kind="variant_genbank_dir",
        source_inputs=tuple(source_inputs_from_manifest(source_manifest, discovered_source_inputs=source_inputs)),
        retired_sources=tuple(source_manifest.get("retired_migration_sources", ())),
    )


def parse_msd_region_genbank_with_replacements(
    source_path: str | Path,
    *,
    replacement_paths: Sequence[str | Path] = (),
    payload_catalog: PayloadBindingCatalog | None = None,
) -> MsdRegionSourceBundle:
    """Parse a base MSD-region GenBank and overlay targeted replacement records."""

    base = parse_msd_region_genbank(source_path, payload_catalog=payload_catalog)
    records_by_variant = {record.variant_id: record for record in base.records}
    skipped_records = list(base.skipped_records)
    replacement_sources: list[dict[str, object]] = []
    for replacement_path in replacement_paths:
        replacement = parse_msd_region_genbank(replacement_path, payload_catalog=payload_catalog)
        if not replacement.records:
            raise MsdRegionIngestError(f"Replacement GenBank has no resolvable retron records: {replacement_path}")
        unknown = sorted({record.variant_id for record in replacement.records} - set(records_by_variant))
        if unknown:
            raise MsdRegionIngestError(
                f"Replacement GenBank contains variants absent from base source {source_path}: {', '.join(unknown)}"
            )
        for record in replacement.records:
            records_by_variant[record.variant_id] = record
        skipped_records.extend(replacement.skipped_records)
        replacement_sources.append(
            {
                "source_path": replacement.source_path,
                "source_sha256": replacement.source_sha256,
                "source_record_count": replacement.source_record_count,
                "included_variant_ids": [record.variant_id for record in replacement.records],
                "skipped_record_count": len(replacement.skipped_records),
            }
        )
    return MsdRegionSourceBundle(
        source_path=base.source_path,
        source_sha256=base.source_sha256,
        source_record_count=base.source_record_count
        + sum(int(source["source_record_count"]) for source in replacement_sources),
        records=tuple(sorted(records_by_variant.values(), key=lambda item: variant_sort_key(item.variant_id))),
        skipped_records=tuple(skipped_records),
        replacement_sources=tuple(replacement_sources),
        source_kind="bulk_migration_genbank_with_replacements",
    )


__all__ = [
    "parse_msd_region_genbank",
    "parse_msd_region_genbank_dir",
    "parse_msd_region_genbank_with_replacements",
]
