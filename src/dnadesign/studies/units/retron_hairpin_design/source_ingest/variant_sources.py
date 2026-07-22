"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/variant_sources.py

Per-variant GenBank source manifests for MSD-region records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from .genbank_utils import sha256_file, variant_id, variant_number, variant_sort_key, write_yaml
from .models import MsdRegionIngestError

VARIANT_SOURCE_MANIFEST = "variant_sources.yaml"


def write_variant_genbank_sources(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    replacement_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Split a bulk migration GenBank into stable one-variant source files."""

    source = Path(source_path).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    variant_dir = root / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    base_records = variant_source_records(source, source_role="bulk_migration_input")
    records_by_variant = dict(base_records["records_by_variant"])
    source_by_variant = {variant_id: "bulk_migration_input" for variant_id in records_by_variant}
    replacement_sources: list[dict[str, object]] = []
    for replacement_path in replacement_paths:
        replacement = Path(replacement_path).expanduser().resolve()
        replacement_records = variant_source_records(replacement, source_role="replacement_overlay")
        unknown = sorted(set(replacement_records["records_by_variant"]) - set(records_by_variant))
        if unknown:
            raise MsdRegionIngestError(
                f"Replacement GenBank contains variants absent from migration source {source}: {', '.join(unknown)}"
            )
        for retron_variant_id, record in replacement_records["records_by_variant"].items():
            records_by_variant[retron_variant_id] = record
            source_by_variant[retron_variant_id] = "replacement_overlay"
        replacement_sources.append(
            {
                "source_file": replacement.name,
                "source_sha256": sha256_file(replacement),
                "source_record_count": replacement_records["source_record_count"],
                "included_variant_ids": sorted(replacement_records["records_by_variant"], key=variant_sort_key),
                "source_role": "replacement_overlay",
            }
        )
    variant_sources: list[dict[str, object]] = []
    for retron_variant_id, record in sorted(records_by_variant.items(), key=lambda item: variant_sort_key(item[0])):
        number = variant_number(retron_variant_id)
        path = variant_dir / f"pes-retron-{int(number):03d}.gb"
        record.annotations.setdefault("molecule_type", "DNA")
        SeqIO.write([record], path, "genbank")
        variant_sources.append(
            {
                "variant_id": retron_variant_id,
                "display_id": f"pES-retron-{number}",
                "source_file": f"variants/{path.name}",
                "source_sha256": sha256_file(path),
                "source_role": source_by_variant[retron_variant_id],
            }
        )
    manifest = {
        "contract": "retron_msd_region_variant_sources_v1",
        "schema_version": 1,
        "source_policy": "per_variant_genbank_sources_are_authority",
        "orientation_rule": "display_msd_5to3_is_reverse_complement_of_genbank_record_sequence",
        "variant_source_dir": "variants",
        "retired_migration_sources": [
            {
                "source_name": source.name,
                "source_sha256": sha256_file(source),
                "source_record_count": base_records["source_record_count"],
                "source_role": "retired_bulk_migration_input",
                "active_source": False,
            }
        ],
        "replacement_sources": replacement_sources,
        "variant_source_count": len(variant_sources),
        "variant_sources": variant_sources,
    }
    manifest_path = root / VARIANT_SOURCE_MANIFEST
    write_yaml(manifest_path, manifest)
    return {
        "manifest_path": manifest_path.as_posix(),
        "variant_source_dir": variant_dir.as_posix(),
        "variant_source_count": len(variant_sources),
        "retired_migration_source_count": 1,
        "replacement_source_count": len(replacement_sources),
    }


def variant_source_records(path: Path, *, source_role: str) -> dict[str, object]:
    records = list(SeqIO.parse(path, "genbank"))
    records_by_variant: dict[str, SeqRecord] = {}
    skipped: list[str] = []
    for record in records:
        retron_variant_id = variant_id(record)
        if retron_variant_id is None:
            skipped.append(record.id)
            continue
        if retron_variant_id in records_by_variant:
            raise MsdRegionIngestError(f"Duplicate MSD-region variant id in {path}: {retron_variant_id}")
        records_by_variant[retron_variant_id] = record
    if source_role == "replacement_overlay" and skipped:
        raise MsdRegionIngestError(f"Replacement source {path} contains unresolved records: {', '.join(skipped)}")
    return {
        "source_record_count": len(records),
        "records_by_variant": records_by_variant,
        "skipped_record_ids": tuple(skipped),
    }


def variant_source_manifest(source_dir: Path) -> dict[str, object]:
    manifest_path = source_dir.parent / VARIANT_SOURCE_MANIFEST
    if not manifest_path.exists():
        return {}
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise MsdRegionIngestError(f"{manifest_path} is not a YAML mapping.")
    if payload.get("source_policy") != "per_variant_genbank_sources_are_authority":
        raise MsdRegionIngestError(
            f"{manifest_path} must declare source_policy=per_variant_genbank_sources_are_authority."
        )
    return payload


def source_inputs_from_manifest(
    manifest: Mapping[str, object],
    *,
    discovered_source_inputs: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    rows = manifest.get("variant_sources")
    if not isinstance(rows, list):
        return tuple(discovered_source_inputs)
    normalized: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise MsdRegionIngestError("variant_sources entries must be YAML mappings.")
        normalized.append(dict(row))
    return tuple(normalized)


__all__ = [
    "VARIANT_SOURCE_MANIFEST",
    "source_inputs_from_manifest",
    "variant_source_manifest",
    "variant_source_records",
    "write_variant_genbank_sources",
]
