"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/bundle_writer.py

Write decomposed MSD-region records and their manifest bundle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .compiler_spec_payload import compiler_spec_payload_from_records
from .genbank_utils import relative_to, write_yaml
from .models import (
    MsdRegionBundleWriteResult,
    MsdRegionComparisonReport,
    MsdRegionSourceBundle,
    NormalizedMsdRegionRecord,
)


def write_msd_region_record_bundle(
    bundle: MsdRegionSourceBundle,
    *,
    output_dir: str | Path,
    comparison_report: MsdRegionComparisonReport | None = None,
) -> MsdRegionBundleWriteResult:
    """Write decomposed records, manifest, compiler spec, and optional review report."""

    root = Path(output_dir).expanduser().resolve()
    records_dir = root / "variants"
    compiler_dir = root / "compiler"
    reports_dir = root / "reports"
    records_dir.mkdir(parents=True, exist_ok=True)
    compiler_dir.mkdir(parents=True, exist_ok=True)
    variant_paths: dict[str, str] = {}
    for record in bundle.records:
        path = records_dir / f"{record.file_stem}.yaml"
        write_yaml(path, record.to_dict())
        variant_paths[record.variant_id] = path.as_posix()
    compiler_spec_path = compiler_dir / "reader_spop_msd_structure_panel_v1.spec.yaml"
    write_yaml(compiler_spec_path, compiler_spec_payload_from_records(bundle.records))
    discrepancy_report_path = _write_comparison_report(comparison_report, reports_dir=reports_dir)
    manifest_path = root / "manifest.yaml"
    write_yaml(
        manifest_path,
        {
            "contract": "retron_msd_region_record_bundle_v1",
            "schema_version": 1,
            "source_policy": _source_policy_for_bundle(bundle),
            "source_kind": bundle.source_kind,
            "source_path": bundle.source_path,
            "source_sha256": bundle.source_sha256,
            "source_inputs": list(bundle.source_inputs),
            "retired_sources": list(bundle.retired_sources),
            "replacement_sources": list(bundle.replacement_sources),
            "source_record_count": bundle.source_record_count,
            "included_record_count": bundle.included_record_count,
            "skipped_record_count": len(bundle.skipped_records),
            "skipped_records": [record.to_dict() for record in bundle.skipped_records],
            "compiler_spec": relative_to(compiler_spec_path, root),
            "discrepancy_report": relative_to(Path(discrepancy_report_path), root)
            if discrepancy_report_path is not None
            else None,
            "records": [_manifest_record(record, variant_paths=variant_paths, root=root) for record in bundle.records],
        },
    )
    return MsdRegionBundleWriteResult(
        output_dir=root.as_posix(),
        manifest_path=manifest_path.as_posix(),
        compiler_spec_path=compiler_spec_path.as_posix(),
        discrepancy_report_path=discrepancy_report_path,
        variant_record_paths=variant_paths,
    )


def _write_comparison_report(
    comparison_report: MsdRegionComparisonReport | None,
    *,
    reports_dir: Path,
) -> str | None:
    if comparison_report is None:
        return None
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "discrepancies.yaml"
    write_yaml(report_path, comparison_report.to_dict())
    return report_path.as_posix()


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


def _source_policy_for_bundle(bundle: MsdRegionSourceBundle) -> str:
    if bundle.source_kind == "variant_genbank_dir":
        return "per_variant_genbank_sources_are_authority"
    return "decomposed_records_are_authority"


__all__ = ["write_msd_region_record_bundle"]
