"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/interfaces/cli/msd_region_ingest.py

Study-owned CLI handler for MSD-region GenBank source ingest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...source_ingest.msd_region_genbank import (
    compare_records_to_existing_sources,
    load_payload_binding_catalog,
    parse_msd_region_genbank_dir,
    write_msd_region_record_bundle,
)
from .io import emit, exit_with_error, format_option

_DEFAULT_STUDY_DIR = Path("docs/studies/retron_hairpin_design")
_DEFAULT_MSD_REGION_RECORD_DIR = Path(
    "docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/retron_msd_structure_panel_v1"
)
_PAYLOAD_BINDING_CATALOG_REL = Path("workbench/ontology/payload_binding_sites.yaml")


def ingest_msd_regions_command(
    source_dir: Path = typer.Option(
        ...,
        "--source-dir",
        exists=True,
        readable=True,
        file_okay=False,
        help="Directory of one-variant GenBank sources. Each file must resolve to exactly one retron MSD variant.",
    ),
    study_dir: Path = typer.Option(_DEFAULT_STUDY_DIR, "--study-dir", help="Retron hairpin study directory."),
    out_dir: Path = typer.Option(
        _DEFAULT_MSD_REGION_RECORD_DIR,
        "--out-dir",
        help="Output directory for decomposed MSD-region records.",
    ),
    compare_existing_output: list[Path] = typer.Option(
        [],
        "--compare-existing-output",
        help="Existing materialized retron-hairpin output root to compare. Repeatable.",
    ),
    payload_binding_catalog: Path | None = typer.Option(
        None,
        "--payload-binding-catalog",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Study-owned payload binding-site catalog. Defaults to workbench/ontology/payload_binding_sites.yaml.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = format_option(output_format)
    try:
        catalog_path = payload_binding_catalog or study_dir / _PAYLOAD_BINDING_CATALOG_REL
        payload_catalog = load_payload_binding_catalog(catalog_path) if catalog_path.exists() else None
        bundle = parse_msd_region_genbank_dir(source_dir, payload_catalog=payload_catalog)
        existing_roots = compare_existing_output or _default_msd_region_comparison_roots(study_dir)
        report = compare_records_to_existing_sources(
            bundle.records,
            existing_roots=existing_roots,
            cap_source_path=study_dir / "compiler/catalog/msd_cap_sources.yaml",
        )
        written = write_msd_region_record_bundle(bundle, output_dir=out_dir, comparison_report=report)
    except (OSError, ValueError) as exc:
        exit_with_error(exc, output_format=format_norm)
    emit(
        {
            "status": "ok",
            "source_kind": bundle.source_kind,
            "source_record_count": bundle.source_record_count,
            "replacement_source_count": len(bundle.replacement_sources),
            "variant_source_input_count": len(bundle.source_inputs),
            "payload_binding_catalog": catalog_path.as_posix() if payload_catalog is not None else None,
            "included_record_count": bundle.included_record_count,
            "skipped_record_count": len(bundle.skipped_records),
            "comparison_count": report.comparison_count,
            "discrepancy_count": report.discrepancy_count,
            "output_dir": written.output_dir,
            "manifest_path": written.manifest_path,
            "compiler_spec_path": written.compiler_spec_path,
            "discrepancy_report_path": written.discrepancy_report_path,
            "variant_record_count": len(written.variant_record_paths),
            "next_step": (
                "Review discrepancies.yaml, then materialize with the emitted compiler spec into a fresh "
                "revisioned workbench/outputs/retron_msd_structure_panel_v1/materialized directory."
            ),
        },
        output_format=format_norm,
    )


def _default_msd_region_comparison_roots(study_dir: Path) -> list[Path]:
    return [
        study_dir / "workbench/outputs/retron-msd-177-194-user-fidelity-20260519",  # pragma: allowlist secret
        study_dir / "workbench/outputs/teto_retained_span_trim_tetr_pwm_elite_v1",
    ]


__all__ = ["ingest_msd_regions_command"]
