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
    RetronMsdCompilerError,
    build_msd_design_reference,
    compile_msd_design_catalog,
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


def _exit_with_error(exc: Exception, *, output_format: str) -> None:
    if output_format == "json":
        _emit({"status": "error", "error": str(exc)}, output_format=output_format)
    else:
        typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1) from exc


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
            "record_count": len(catalog.records),
            "records": [record.model_dump(mode="json") for record in catalog.records],
        },
        output_format=format_norm,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
