"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/retron_hairpin_design/cli.py

Study-owned CLI for Retron MSD design identifiers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import typer

from .compiler import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_README_FILENAME,
    COMPOSITION_CONFIG_DIRNAME,
    MANIFEST_DIRNAME,
    REFERENCE_DIRNAME,
    REFERENCE_INDEX_FILENAME,
    SEQUENCE_INDEX_FILENAME,
    SEQUENCE_MANIFEST_FILENAME,
    VARIANT_DIRNAME,
    RetronMsdCompilerError,
    build_msd_design_reference,
    compile_msd_design_catalog,
    materialize_msd_design_artifacts,
    write_msd_design_catalog,
)
from .msd_ids import MsdIdError
from .registry import RetronMsdRegistryError

_DEFAULT_STUDY_DIR = Path("docs/studies/retron_hairpin_design")

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="Lint and compile Retron MSD construct labels into design-reference contracts.",
)


def _format_option(output_format: str) -> str:
    format_norm = str(output_format or "").strip().lower()
    if format_norm not in {"text", "json"}:
        raise typer.BadParameter("Output format must be text or json.")
    return format_norm


def _emit(payload: dict[str, Any], *, output_format: str) -> None:
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"status: {payload.get('status')}")
    if payload.get("reference") is not None:
        reference = payload["reference"]
        typer.echo(f"msd_design_id: {reference.get('msd_design_id')}")
        typer.echo(f"construct_id: {reference.get('construct_id')}")
    if payload.get("catalog_path") is not None:
        typer.echo(f"catalog_path: {payload['catalog_path']}")
        typer.echo(f"record_count: {payload.get('record_count')}")
    if payload.get("output_dir") is not None:
        typer.echo(f"output_dir: {payload['output_dir']}")
    if payload.get("references_dir") is not None:
        typer.echo(f"references_dir: {payload['references_dir']}")
    if payload.get("index_path") is not None:
        typer.echo(f"index_path: {payload['index_path']}")
    if payload.get("manifest_path") is not None:
        typer.echo(f"manifest_path: {payload['manifest_path']}")
    if payload.get("readme_path") is not None:
        typer.echo(f"readme_path: {payload['readme_path']}")
    if payload.get("sequence_manifest_path") is not None:
        typer.echo(f"sequence_manifest_path: {payload['sequence_manifest_path']}")
    if payload.get("sequence_index_path") is not None:
        typer.echo(f"sequence_index_path: {payload['sequence_index_path']}")
    if payload.get("variants_dir") is not None:
        typer.echo(f"variants_dir: {payload['variants_dir']}")
    if payload.get("composition_configs_dir") is not None:
        typer.echo(f"composition_configs_dir: {payload['composition_configs_dir']}")
    if payload.get("finder_open") is not None:
        typer.echo(f"finder_open: {payload['finder_open']}")
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            typer.echo(f"warning: {warning}")
    if payload.get("next_step") is not None:
        typer.echo(f"next_step: {payload['next_step']}")


def _exit_with_error(exc: Exception, *, output_format: str) -> None:
    next_step = _next_step_for_error(exc)
    if output_format == "json":
        _emit(
            {
                "status": "error",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "next_step": next_step,
            },
            output_format=output_format,
        )
    else:
        typer.echo(f"error: {exc}", err=True)
        typer.echo(f"next_step: {next_step}", err=True)
    raise typer.Exit(code=1) from exc


def _next_step_for_error(exc: Exception) -> str:
    message = str(exc)
    if "provided profile" in message:
        return "Correct the declared -MWX profile or omit it so the compiler derives S3/S2/S1/S0 from the bases."
    if "S0" in message:
        return (
            "Route the left/right base feasibility question to scar-nick before compiling; the compiler requires S0=M."
        )
    if "Unknown cap" in message:
        return (
            "Route missing cap or shortening constraints to Snapback, "
            "or add the validated cap to msd_design_registry.yaml."
        )
    if "Unknown payload" in message:
        return "Add the validated payload to msd_design_registry.yaml before compiling a frozen design reference."
    if "registry" in message:
        return (
            "Open docs/studies/retron_hairpin_design/msd_design_registry.yaml "
            "and fix the registry before rerunning lint."
        )
    if "Duplicate construct label" in message:
        return "Deduplicate the input labels, then rerun compile with the same explicit --out-dir."
    if "Duplicate MSD design reference filename" in message:
        return "Deduplicate equivalent MSD design IDs before writing a catalog bundle."
    if "Legacy MSD compiler output layout" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove the old generated assets directory."
    if "Unexpected MSD materialize output entries" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove stale flat materialize output first."
    if "Unexpected MSD compiler output entries" in message or "Stale MSD design reference output" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove unrelated generated output before compiling."
    if "MSD sequence artifact generation requires concrete sequence subcomponents" in message:
        return (
            "Provide literal subcomponents with --payload-sequence ID=ACGT and --cap-sequence ID=ACGT, "
            "or route missing cap/shortening inputs to Snapback before generating GenBank/PNG artifacts."
        )
    if "Stale MSD sequence output" in message or "Stale MSD composition config output" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove stale generated sequence outputs first."
    return "Run lint on one complete MSD label first; route missing biological constraints before generating a catalog."


def _lint_next_step() -> str:
    return "Input is complete; run compile with an explicit --out-dir when a design-reference catalog is needed."


def _compile_next_step() -> str:
    return (
        "Catalog bundle emitted with flat references; run materialize with explicit payload/cap sequences "
        "when one GenBank/PNG sequence bundle per MSD design is needed."
    )


def _materialize_warnings(variants: list[dict[str, Any]]) -> list[str]:
    folding_warning_count = sum(1 for variant in variants if variant.get("folding_status") != "ok")
    if folding_warning_count == 0:
        return []
    statuses = sorted(
        {str(variant.get("folding_status")) for variant in variants if variant.get("folding_status") != "ok"}
    )
    return [
        "Folding was attempted for every variant, but "
        f"{folding_warning_count} variant(s) reported {', '.join(statuses)}. "
        "Install ViennaRNA RNAfold or run on a PATH that exposes RNAfold to get structure predictions; "
        "no fallback prediction was used."
    ]


def _materialize_next_step(out_dir: Path, *, warnings: list[str]) -> str:
    if warnings:
        return (
            "Single-unit MSD sequence bundle emitted with GenBank, FASTA/CSV, and plot/status artifacts; "
            f"open {out_dir.as_posix()} or inspect manifest/sequence_index.tsv for folding status."
        )
    return (
        "Single-unit MSD sequence bundle emitted with GenBank, FASTA/CSV, folding, and plot artifacts; "
        f"open {out_dir.as_posix()} or use manifest/sequence_index.tsv for programmatic handoff."
    )


def _collect_labels(ids: list[str], input_file: Path | None) -> list[str]:
    labels = [item.strip() for item in ids if item.strip()]
    if input_file is not None:
        labels.extend(_read_input_labels(input_file))
    if not labels:
        raise RetronMsdCompilerError("Provide at least one --id or an --input file with construct labels.")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise RetronMsdCompilerError(f"Duplicate construct label(s): {', '.join(duplicates)}")
    return labels


def _read_input_labels(input_file: Path) -> list[str]:
    path = input_file.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".csv", ".tsv", ".tab"}:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        rows = csv.DictReader(text.splitlines(), delimiter=delimiter)
        if rows.fieldnames is None:
            raise RetronMsdCompilerError(f"Input file has no header row: {path}")
        for field in ("construct_label", "design_id", "id"):
            if field in rows.fieldnames:
                return [str(row.get(field, "")).strip() for row in rows if str(row.get(field, "")).strip()]
        raise RetronMsdCompilerError("CSV/TSV input must include construct_label, design_id, or id column.")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _sequence_override_map(values: list[str], *, label: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        key, separator, value = text.partition("=")
        if separator != "=" or not key.strip() or not value.strip():
            raise RetronMsdCompilerError(f"{label} override must be ID=SEQUENCE.")
        overrides[key.strip()] = value.strip()
    return overrides


@app.command("lint")
def lint_command(
    label: str = typer.Option(..., "--id", help="Retron MSD construct label to parse and validate."),
    study_dir: Path = typer.Option(_DEFAULT_STUDY_DIR, "--study-dir", help="Retron hairpin study directory."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = _format_option(output_format)
    try:
        reference = build_msd_design_reference(label, study_dir=study_dir)
    except (MsdIdError, RetronMsdRegistryError, RetronMsdCompilerError, OSError, ValueError) as exc:
        _exit_with_error(exc, output_format=format_norm)
    _emit(
        {
            "status": "ok",
            "reference": reference.model_dump(mode="json"),
            "next_step": _lint_next_step(),
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
        help="Snapback-cap sequence override as ID=ACGT. Repeat for each cap ID.",
    ),
    render_format: list[str] = typer.Option(
        ["png"],
        "--render-format",
        help="BaseRender component-span export format. Repeat for png/svg/pdf.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = _format_option(output_format)
    try:
        labels = _collect_labels(ids, input_file)
        catalog = compile_msd_design_catalog(labels, study_dir=study_dir)
        result = materialize_msd_design_artifacts(
            catalog,
            out_dir=out_dir,
            payload_sequences=_sequence_override_map(payload_sequence, label="payload sequence"),
            cap_sequences=_sequence_override_map(cap_sequence, label="cap sequence"),
            render_formats=render_format,
        )
    except (MsdIdError, RetronMsdRegistryError, RetronMsdCompilerError, OSError, ValueError) as exc:
        _exit_with_error(exc, output_format=format_norm)
    warnings = _materialize_warnings(result.variants)
    _emit(
        {
            "status": "ok",
            "catalog_path": str(result.bundle_root / MANIFEST_DIRNAME / "msd_design_catalog_v1.json"),
            "output_dir": str(result.bundle_root),
            "references_dir": str(result.bundle_root / MANIFEST_DIRNAME / REFERENCE_DIRNAME),
            "index_path": str(result.bundle_root / MANIFEST_DIRNAME / REFERENCE_INDEX_FILENAME),
            "manifest_path": str(result.bundle_root / MANIFEST_DIRNAME / BUNDLE_MANIFEST_FILENAME),
            "readme_path": str(result.bundle_root / BUNDLE_README_FILENAME),
            "sequence_manifest_path": str(result.bundle_root / MANIFEST_DIRNAME / SEQUENCE_MANIFEST_FILENAME),
            "sequence_index_path": str(result.bundle_root / MANIFEST_DIRNAME / SEQUENCE_INDEX_FILENAME),
            "variants_dir": str(result.bundle_root / VARIANT_DIRNAME),
            "composition_configs_dir": str(result.bundle_root / MANIFEST_DIRNAME / COMPOSITION_CONFIG_DIRNAME),
            "record_count": len(result.catalog.records),
            "variants": result.variants,
            "records": [record.model_dump(mode="json") for record in result.catalog.records],
            "finder_open": f"open {result.bundle_root.as_posix()}",
            "warnings": warnings,
            "next_step": _materialize_next_step(result.bundle_root, warnings=warnings),
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
    study_dir: Path = typer.Option(_DEFAULT_STUDY_DIR, "--study-dir", help="Retron hairpin study directory."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory for emitted design-reference catalog."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = _format_option(output_format)
    try:
        labels = _collect_labels(ids, input_file)
        catalog = compile_msd_design_catalog(labels, study_dir=study_dir)
        catalog_path = write_msd_design_catalog(catalog, out_dir=out_dir)
    except (MsdIdError, RetronMsdRegistryError, RetronMsdCompilerError, OSError, ValueError) as exc:
        _exit_with_error(exc, output_format=format_norm)
    _emit(
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
            "next_step": _compile_next_step(),
        },
        output_format=format_norm,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
