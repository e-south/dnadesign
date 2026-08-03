"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/bundle_manifest.py

Project normalized MSD-region records into one portable bundle manifest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .genbank_utils import relative_to
from .models import MsdRegionSourceBundle, NormalizedMsdRegionRecord


def build_msd_region_manifest_payload(
    bundle: MsdRegionSourceBundle,
    *,
    bundle_root: Path,
    variant_paths: dict[str, str],
    compiler_spec_path: Path,
    discrepancy_report_path: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract": "retron_msd_region_record_bundle_v1",
        "schema_version": 1,
    }
    payload.update(
        {
            "source_policy": _source_policy(bundle.source_kind),
            "source_kind": bundle.source_kind,
            "source_path": _portable_source_ref(bundle.source_path, bundle_root=bundle_root),
            "source_sha256": bundle.source_sha256,
            "source_inputs": list(bundle.source_inputs),
            "retired_sources": list(bundle.retired_sources),
            "replacement_sources": list(bundle.replacement_sources),
            "source_record_count": bundle.source_record_count,
            "included_record_count": bundle.included_record_count,
            "skipped_record_count": len(bundle.skipped_records),
            "skipped_records": [record.to_dict() for record in bundle.skipped_records],
            "compiler_spec": relative_to(compiler_spec_path, bundle_root),
            "discrepancy_report": (
                relative_to(Path(discrepancy_report_path), bundle_root) if discrepancy_report_path is not None else None
            ),
            "records": [
                _manifest_record(record, variant_paths=variant_paths, root=bundle_root) for record in bundle.records
            ],
        }
    )
    return payload


def _manifest_record(
    record: NormalizedMsdRegionRecord,
    *,
    variant_paths: dict[str, str],
    root: Path,
) -> dict[str, object]:
    return {
        "variant_id": record.variant_id,
        "display_id": record.display_id,
        "record": relative_to(Path(variant_paths[record.variant_id]), root),
        "annotation_status": record.annotation_status,
        "annotation_warning_count": len(record.annotation_warnings),
        "annotation_warnings": [warning.to_dict() for warning in record.annotation_warnings],
        "annotation_note_count": len(record.annotation_notes),
        "annotation_notes": [note.to_dict() for note in record.annotation_notes],
        "pairing_segments": [segment.to_dict() for segment in record.pairing_segments],
        "payload_binding_sites": [site.to_dict() for site in record.payload_binding_sites],
        "msd_sequence_sha256": record.msd_sequence_sha256,
    }


def _source_policy(source_kind: str) -> str:
    if source_kind == "variant_genbank_dir":
        return "per_variant_genbank_sources_are_authority"
    return "decomposed_records_are_authority"


def _portable_source_ref(source_path: str, *, bundle_root: Path) -> str:
    source = Path(source_path).expanduser().resolve()
    try:
        return source.relative_to(bundle_root).as_posix()
    except ValueError:
        return source.name


__all__ = ["build_msd_region_manifest_payload"]
