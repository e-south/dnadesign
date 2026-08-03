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

from .bundle_manifest import build_msd_region_manifest_payload
from .compiler_spec_payload import compiler_spec_payload_from_records
from .genbank_utils import write_yaml
from .models import (
    MsdRegionBundleWriteResult,
    MsdRegionComparisonReport,
    MsdRegionSourceBundle,
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
    variant_paths = {
        record.variant_id: (records_dir / f"{record.file_stem}.yaml").as_posix() for record in bundle.records
    }
    variant_payloads = {record.variant_id: record.to_dict() for record in bundle.records}
    compiler_spec_path = compiler_dir / "retron_msd_structure_panel_v1.spec.yaml"
    compiler_spec_payload = compiler_spec_payload_from_records(bundle.records)
    discrepancy_report_path = (reports_dir / "discrepancies.yaml").as_posix() if comparison_report is not None else None
    manifest_path = root / "manifest.yaml"
    manifest_payload = build_msd_region_manifest_payload(
        bundle,
        bundle_root=root,
        variant_paths=variant_paths,
        compiler_spec_path=compiler_spec_path,
        discrepancy_report_path=discrepancy_report_path,
    )

    records_dir.mkdir(parents=True, exist_ok=True)
    compiler_dir.mkdir(parents=True, exist_ok=True)
    for record in bundle.records:
        write_yaml(Path(variant_paths[record.variant_id]), variant_payloads[record.variant_id])
    write_yaml(compiler_spec_path, compiler_spec_payload)
    _write_comparison_report(comparison_report, reports_dir=reports_dir)
    write_yaml(manifest_path, manifest_payload)
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


__all__ = ["write_msd_region_record_bundle"]
