"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/cli/app.py

CLI wiring for app Permuter CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.traceback import install as rich_tb

from dnadesign.permuter.src.core.logging_setup import configure_logging
from dnadesign.permuter.src.plots.registry import supported_plot_ids

from . import evaluate as eval_cmd
from . import export as export_cmd
from . import inspect as inspect_cmd
from . import plot as plot_cmd
from . import run as run_cmd
from . import validate as validate_cmd
from . import workspace as workspace_cmd

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Permuter — mutate biological sequences, score them with pluggable evaluators, "
        "and analyze/export results.\n\n"
        "\b\nTypical workflow:\n"
        "  • permuter run      - generate variants for a reference using a protocol\n"
        "  • permuter evaluate - append metric columns into the same Parquet\n"
        "  • permuter plot     - write plot artifacts alongside the dataset\n"
        "  • permuter export   - optional CSV/JSONL export\n"
        "  • permuter validate - structural & integrity checks\n\n"
        "\b\nNotes:\n"
        "  • --workspace accepts a workspace directory, config.yaml path, or scope id.\n"
        "  • --data accepts a dataset directory OR a records.parquet file.\n"
        "  • ${WORKSPACE_DIR}, ${WORKSPACES_DIR}, ${PERMUTER_RESOURCE_DIR}, env vars, and ~ are expanded.\n"
        "  • Output defaults to the workspace's configured outputs/ path.\n"
        "    No silent fallbacks: if unwritable, use $PERMUTER_OUTPUT_ROOT or --out."
    ),
)
console = Console()
# Keep CLI failures readable without dumping local sequences or workspace payloads.
rich_tb(show_locals=False)
app.add_typer(workspace_cmd.app, name="workspace")


@app.callback()
def _root(verbose: int = typer.Option(0, "--verbose", "-v", count=True)):
    configure_logging(verbose)


@app.command(
    "run",
    help=(
        "Generate variants for a single reference using the configured protocol.\n"
        "Writes one dataset directory per reference with records.parquet, REF.fa and plots/."
    ),
)
def run(
    workspace: str = typer.Option(..., "--workspace", "-w", help="Workspace directory, config.yaml path, or scope id."),
    ref: str = typer.Option(
        None,
        "--ref",
        help="Reference name (row in refs CSV). Required if CSV has multiple rows.",
    ),
    out: Path = typer.Option(None, "--out", "-o", help="Output root directory (default: workspace output.dir)"),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Replace existing dataset (records.parquet) if present.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a single machine-readable JSON result."),
):
    run_cmd.run(workspace=workspace, ref=ref, out=out, overwrite=overwrite, as_json=as_json)


@app.command(
    "evaluate",
    help=(
        "Append metric columns to records.parquet using one or more evaluators. "
        "Use --with id:evaluator[:metric] or --workspace to read evaluate.metrics[]."
    ),
)
def evaluate(
    data: Path = typer.Option(
        None,
        "--data",
        "-d",
        help="Path to records.parquet OR dataset directory (optional if --workspace/--ref given)",
    ),
    with_spec: List[str] = typer.Option(
        None,
        "--with",
        help="Repeatable: id:evaluator[:metric] (e.g., llr:evo2_llr:log_likelihood_ratio)",
    ),
    metric: List[str] = typer.Option(None, "--metric", help="Convenience: metric ids scored by placeholder evaluator"),
    workspace: str = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory, config.yaml path, or scope id (used if --data omitted)",
    ),
    ref: str = typer.Option(None, "--ref", help="Reference name from refs CSV (used if --data omitted)"),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Override output root when deriving dataset from --workspace/--ref",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a single machine-readable JSON result."),
):
    eval_cmd.evaluate(
        data=data,
        metric_ids=list(metric or []),
        with_spec=list(with_spec or []),
        workspace=workspace,
        ref=ref,
        out=out,
        as_json=as_json,
    )


@app.command(
    "plot",
    help=(
        "Generate plot artifacts for a dataset. "
        "Use --metric-id (or workspace plot.metric_id) to choose a metric when multiple exist. "
        "Repeat --which to draw multiple plots."
    ),
)
def plot(
    data: Path = typer.Option(None, "--data", "-d", help="Path to records.parquet OR dataset directory"),
    workspace: str = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory, config.yaml path, or scope id (used if --data omitted)",
    ),
    ref: str = typer.Option(None, "--ref", help="Reference name from refs CSV (used with --workspace)"),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Override output root when deriving dataset from --workspace/--ref",
    ),
    which: List[str] = typer.Option(
        None,
        "--which",
        help=f"Plot id to generate (repeat for multiple). Allowed: {', '.join(supported_plot_ids())}",
    ),
    metric_id: str = typer.Option(None, "--metric-id", help="Metric id to plot (e.g., llr_mean, llr_sum)"),
    width: float = typer.Option(None, "--width", help="Figure width (inches)"),
    height: float = typer.Option(None, "--height", help="Figure height (inches)"),
    font_scale: float = typer.Option(None, "--font-scale", help="Multiply all font sizes"),
    emit_summaries: Optional[bool] = typer.Option(
        None,
        "--emit-summaries/--no-emit-summaries",
        help="Emit analysis summaries (e.g., AA LLR Top/Bottom CSV) during plotting (default: on).",
    ),
    list_plots: bool = typer.Option(False, "--list", help="List supported plot contracts."),
    describe: str = typer.Option(None, "--describe", help="Describe one supported plot contract."),
    as_json: bool = typer.Option(False, "--json", help="Emit a single machine-readable JSON result."),
):
    try:
        plot_cmd.plot(
            data=data,
            workspace=workspace,
            ref=ref,
            out=out,
            which=which,
            metric_id=metric_id,
            width=width,
            height=height,
            font_scale=font_scale,
            emit_summaries=emit_summaries,
            list_plots=list_plots,
            describe=describe,
            as_json=as_json,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command(
    "export",
    help="Export a dataset to CSV or JSONL while preserving column names.",
)
def export(
    data: Path = typer.Option(..., "--data", "-d", exists=True, readable=True),
    fmt: str = typer.Option("csv", "--fmt", help="csv|jsonl"),
    out: Path = typer.Option(..., "--out", "-o", help="Output file path"),
):
    export_cmd.export_(data=data, fmt=fmt, out=out)


@app.command(
    "validate",
    help="Validate USR core columns, ID integrity, and required permuter columns (strict mode).",
)
def validate(
    data: Path = typer.Option(..., "--data", "-d", exists=True, readable=True),
    strict: bool = typer.Option(False, "--strict"),
    record: bool = typer.Option(
        False,
        "--record/--no-record",
        help="Append this validation command to RECORD.md.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a single machine-readable JSON result."),
):
    validate_cmd.validate(data=data, strict=strict, record=record, as_json=as_json)


@app.command(
    "inspect",
    help="Print a small summary table and the head of the dataset for quick inspection.",
)
def inspect(
    data: Path = typer.Option(..., "--data", "-d", exists=True, readable=True),
    head: int = typer.Option(5, "--head", "-n"),
    record: bool = typer.Option(
        False,
        "--record/--no-record",
        help="Append this inspection command to RECORD.md.",
    ),
):
    inspect_cmd.inspect_(data=data, head=head, record=record)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
