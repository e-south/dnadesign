"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/interfaces/cli/app.py

Study-owned CLI for Retron MSD design identifiers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...catalog.compiler_spec import MsdCompilerSpecError, load_msd_compiler_spec
from ...catalog.msd_ids import MsdIdError
from ...catalog.registry import RetronMsdRegistryError
from ...compiler.catalog_bundle import write_msd_design_catalog
from ...compiler.exceptions import RetronMsdCompilerError
from ...compiler.materialization import materialize_msd_design_artifacts
from ...compiler.references import build_msd_design_reference, compile_msd_design_catalog
from ...outputs.layout import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_README_FILENAME,
    CATALOG_FILENAME,
    COMPOSITION_CONFIG_DIRNAME,
    MANIFEST_BUNDLE_DIRNAME,
    MANIFEST_CATALOG_DIRNAME,
    MANIFEST_CONFIGS_DIRNAME,
    MANIFEST_DIRNAME,
    MANIFEST_INDEXES_DIRNAME,
    REFERENCE_DIRNAME,
    REFERENCE_INDEX_FILENAME,
    VARIANT_DIRNAME,
)
from .inputs import collect_labels, merge_sequence_maps, reject_mixed_design_sources, sequence_override_map
from .io import emit, exit_with_error, format_option
from .messages import compile_next_step, lint_next_step, materialize_next_step, materialize_warnings

_DEFAULT_STUDY_DIR = Path("docs/studies/retron_hairpin_design")

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="Lint and compile Retron MSD construct labels into design-reference contracts.",
)


@app.command("lint")
def lint_command(
    label: str | None = typer.Option(None, "--id", help="Retron MSD construct label to parse and validate."),
    spec_file: Path | None = typer.Option(
        None,
        "--spec",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Typed retron_msd_compiler_spec_v1 YAML/JSON file to parse and validate.",
    ),
    study_dir: Path = typer.Option(_DEFAULT_STUDY_DIR, "--study-dir", help="Retron hairpin study directory."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = format_option(output_format)
    try:
        if spec_file is not None:
            if label is not None:
                raise RetronMsdCompilerError("Use either lint --spec or lint --id, not both.")
            resolved = load_msd_compiler_spec(spec_file, study_dir=study_dir)
            emit(
                {
                    "status": "ok",
                    "record_count": len(resolved.catalog.records),
                    "records": [record.model_dump(mode="json") for record in resolved.catalog.records],
                    "next_step": lint_next_step(),
                },
                output_format=format_norm,
            )
            return
        if label is None:
            raise RetronMsdCompilerError("Provide --id or --spec for lint.")
        reference = build_msd_design_reference(label, study_dir=study_dir)
    except (
        MsdIdError,
        RetronMsdRegistryError,
        RetronMsdCompilerError,
        MsdCompilerSpecError,
        OSError,
        ValueError,
    ) as exc:
        exit_with_error(exc, output_format=format_norm)
    emit(
        {
            "status": "ok",
            "reference": reference.model_dump(mode="json"),
            "next_step": lint_next_step(),
        },
        output_format=format_norm,
    )


@app.command("materialize")
def materialize_command(
    ids: list[str] = typer.Option([], "--id", help="Retron MSD construct label. May be supplied more than once."),
    input_file: Path | None = typer.Option(
        None,
        "--input",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Plain-text, CSV, or TSV list of Retron MSD construct labels.",
    ),
    spec_file: Path | None = typer.Option(
        None,
        "--spec",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Typed retron_msd_compiler_spec_v1 YAML/JSON file with labels/designs and optional sequences.",
    ),
    study_dir: Path = typer.Option(_DEFAULT_STUDY_DIR, "--study-dir", help="Retron hairpin study directory."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Transient directory for the sequence artifact bundle."),
    payload_sequence: list[str] = typer.Option(
        [],
        "--payload-sequence",
        help="Payload/target sequence override as ID=ACGT. Repeat for each payload ID.",
    ),
    cap_sequence: list[str] = typer.Option(
        [],
        "--cap-sequence",
        help="Cap/foldback segment sequence override as ID=ACGT. Repeat for each cap ID.",
    ),
    render_format: list[str] = typer.Option(
        ["png"],
        "--render-format",
        help="BaseRender component-span export format. Repeat for png/svg/pdf.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = format_option(output_format)
    try:
        reject_mixed_design_sources(ids=ids, input_file=input_file, spec_file=spec_file)
        cli_payload_sequences = sequence_override_map(payload_sequence, label="payload sequence")
        cli_cap_sequences = sequence_override_map(cap_sequence, label="cap sequence")
        if spec_file is None:
            labels = collect_labels(ids, input_file)
            catalog = compile_msd_design_catalog(labels, study_dir=study_dir)
            resolved_payload_sequences: dict[str, str] = {}
            resolved_cap_sequences: dict[str, str] = {}
        else:
            resolved = load_msd_compiler_spec(spec_file, study_dir=study_dir)
            catalog = resolved.catalog
            resolved_payload_sequences = resolved.payload_sequences
            resolved_cap_sequences = resolved.cap_sequences
        result = materialize_msd_design_artifacts(
            catalog,
            out_dir=out_dir,
            payload_sequences=merge_sequence_maps(
                resolved_payload_sequences,
                cli_payload_sequences,
                label="payload",
            ),
            cap_sequences=merge_sequence_maps(resolved_cap_sequences, cli_cap_sequences, label="cap"),
            render_formats=render_format,
        )
    except (
        MsdIdError,
        RetronMsdRegistryError,
        RetronMsdCompilerError,
        MsdCompilerSpecError,
        OSError,
        ValueError,
    ) as exc:
        exit_with_error(exc, output_format=format_norm)
    warnings = materialize_warnings(result.variants)
    emit(
        {
            "status": "ok",
            "catalog_path": str(result.bundle_root / MANIFEST_DIRNAME / MANIFEST_CATALOG_DIRNAME / CATALOG_FILENAME),
            "output_dir": str(result.bundle_root),
            "references_dir": str(result.bundle_root / MANIFEST_DIRNAME / MANIFEST_CATALOG_DIRNAME / REFERENCE_DIRNAME),
            "index_path": str(
                result.bundle_root / MANIFEST_DIRNAME / MANIFEST_INDEXES_DIRNAME / REFERENCE_INDEX_FILENAME
            ),
            "manifest_path": str(
                result.bundle_root / MANIFEST_DIRNAME / MANIFEST_BUNDLE_DIRNAME / BUNDLE_MANIFEST_FILENAME
            ),
            "readme_path": str(result.bundle_root / BUNDLE_README_FILENAME),
            "sequence_manifest_path": str(result.manifest_path),
            "sequence_index_path": str(result.index_path),
            "variants_dir": str(result.bundle_root / VARIANT_DIRNAME),
            "composition_configs_dir": str(
                result.bundle_root / MANIFEST_DIRNAME / MANIFEST_CONFIGS_DIRNAME / COMPOSITION_CONFIG_DIRNAME
            ),
            "record_count": len(result.catalog.records),
            "variants": result.variants,
            "records": [record.model_dump(mode="json", exclude_none=True) for record in result.catalog.records],
            "finder_open": f"open {result.bundle_root.as_posix()}",
            "warnings": warnings,
            "next_step": materialize_next_step(result.bundle_root, warnings=warnings),
        },
        output_format=format_norm,
    )


@app.command("compile")
def compile_command(
    ids: list[str] = typer.Option([], "--id", help="Retron MSD construct label. May be supplied more than once."),
    input_file: Path | None = typer.Option(
        None,
        "--input",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Plain-text, CSV, or TSV list of Retron MSD construct labels.",
    ),
    spec_file: Path | None = typer.Option(
        None,
        "--spec",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Typed retron_msd_compiler_spec_v1 YAML/JSON file with labels or explicit designs.",
    ),
    study_dir: Path = typer.Option(_DEFAULT_STUDY_DIR, "--study-dir", help="Retron hairpin study directory."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory for emitted design-reference catalog."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = format_option(output_format)
    try:
        reject_mixed_design_sources(ids=ids, input_file=input_file, spec_file=spec_file)
        if spec_file is None:
            labels = collect_labels(ids, input_file)
            catalog = compile_msd_design_catalog(labels, study_dir=study_dir)
        else:
            catalog = load_msd_compiler_spec(spec_file, study_dir=study_dir).catalog
        catalog_path = write_msd_design_catalog(catalog, out_dir=out_dir)
    except (
        MsdIdError,
        RetronMsdRegistryError,
        RetronMsdCompilerError,
        MsdCompilerSpecError,
        OSError,
        ValueError,
    ) as exc:
        exit_with_error(exc, output_format=format_norm)
    emit(
        {
            "status": "ok",
            "catalog_path": str(catalog_path),
            "output_dir": str(catalog_path.parent),
            "references_dir": str(catalog_path.parent / REFERENCE_DIRNAME),
            "index_path": str(catalog_path.parent / REFERENCE_INDEX_FILENAME),
            "manifest_path": str(catalog_path.parent / BUNDLE_MANIFEST_FILENAME),
            "readme_path": str(catalog_path.parent / BUNDLE_README_FILENAME),
            "record_count": len(catalog.records),
            "records": [record.model_dump(mode="json") for record in catalog.records],
            "next_step": compile_next_step(),
        },
        output_format=format_norm,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
