"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/compose.py

construct compose command implementation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
from pathlib import Path

import typer

from ...composition.review import publish_composition_review_svg
from ...composition.runtime import run_linear_ssdna_composition, summarize_linear_ssdna_composition
from ...contracts.errors import ConstructError
from ._errors import exit_with_error
from ._format import echo_json, validate_output_format

compose_app = typer.Typer(no_args_is_help=True, help="Compose generic linear ssDNA products from segment specs.")


@compose_app.command("validate")
def validate_composition(
    config: Path = typer.Option(..., "--config", exists=True, readable=True, help="Linear ssDNA composition YAML."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_requested = str(output_format or "").strip().lower()
    try:
        format_norm = validate_output_format(output_format)
        summary = summarize_linear_ssdna_composition(config)
    except (ConstructError, OSError) as exc:
        exit_with_error(exc, code=1, output_format=format_requested)
    if format_norm == "json":
        echo_json(
            {
                "status": "ok",
                "composition_id": summary.composition_id,
                "unit_count": summary.unit_count,
                "expanded_copy_count": summary.expanded_copy_count,
                "sequence_length": summary.sequence_length,
            }
        )
        return
    typer.echo(f"Composition config OK: {config}")
    typer.echo(f"composition_id: {summary.composition_id}")
    typer.echo(f"unit_count: {summary.unit_count}")
    typer.echo(f"expanded_copy_count: {summary.expanded_copy_count}")
    typer.echo(f"sequence_length: {summary.sequence_length}")


@compose_app.command("run")
def run_composition(
    config: Path = typer.Option(..., "--config", exists=True, readable=True, help="Linear ssDNA composition YAML."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_requested = str(output_format or "").strip().lower()
    try:
        format_norm = validate_output_format(output_format)
        result = run_linear_ssdna_composition(config)
    except (ConstructError, OSError) as exc:
        exit_with_error(exc, code=1, output_format=format_requested)
    if format_norm == "json":
        echo_json(
            {
                "status": "ok",
                "composition": result,
                "artifacts": _composition_artifact_hints(result.artifact_bundle),
            }
        )
        return
    typer.echo(
        "Composition run complete: "
        f"composition={result.composition_id} length={result.sequence_length} sha256={result.sequence_sha256}"
    )
    typer.echo(f"artifact_bundle: {result.artifact_bundle}")
    typer.echo(f"manifest: {result.manifest_path}")
    hints = _composition_artifact_hints(result.artifact_bundle)
    typer.echo(f"genbank: {hints['genbank']}")
    typer.echo(f"finder_reveal: {hints['finder_reveal']}")


@compose_app.command("review")
def review_composition(
    bundle: Path = typer.Option(
        ...,
        "--bundle",
        exists=True,
        file_okay=False,
        readable=True,
        writable=True,
        help="Linear ssDNA composition artifact bundle.",
    ),
    nucleotide_font_size_px: float = typer.Option(
        6.0,
        "--nucleotide-font-size-px",
        min=0.1,
        help="Effective nucleotide font size to match across review panels.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_requested = str(output_format or "").strip().lower()
    try:
        format_norm = validate_output_format(output_format)
        manifest = publish_composition_review_svg(
            bundle,
            target_nucleotide_font_size_px=nucleotide_font_size_px,
        )
    except (ConstructError, OSError) as exc:
        exit_with_error(exc, code=1, output_format=format_requested)
    if format_norm == "json":
        echo_json(
            {
                "status": "ok",
                "review": manifest,
            }
        )
        return
    typer.echo(f"Composition review complete: review_id={manifest.review_id}")
    typer.echo(f"review_svg: {Path(bundle) / manifest.artifacts.review_svg}")
    typer.echo(f"review_png: {Path(bundle) / manifest.artifacts.review_png}")


def _composition_artifact_hints(artifact_bundle: Path) -> dict[str, str]:
    genbank_path = Path(artifact_bundle) / "sequence.gb"
    return {
        "genbank": genbank_path.as_posix(),
        "finder_reveal": f"open -R {shlex.quote(genbank_path.as_posix())}",
    }
