"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/app.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.traceback import install as rich_tb

# Import sibling CLI modules via relative imports
from . import evaluate as eval_cmd
from . import export as export_cmd
from . import inspect as inspect_cmd
from . import plot as plot_cmd
from . import run as run_cmd
from . import validate as validate_cmd

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Permuter — mutate biological sequences, score them with pluggable evaluators, "
        "and analyze/export results.\n\n"
        "\b\nTypical workflow:\n"
        "  • permuter run      - generate variants for a reference using a protocol\n"
        "  • permuter evaluate - append metric columns into the same Parquet\n"
        "  • permuter plot     - write PNG plots alongside the dataset\n"
        "  • permuter export   - optional CSV/JSONL export\n"
        "  • permuter validate - structural & integrity checks\n\n"
        "\b\nNotes:\n"
        "  • --job accepts a path or PRESET NAME (search: $PERMUTER_JOBS, CWD[/jobs], repo, package jobs/).\n"
        "  • --data accepts a dataset directory OR a records.parquet file.\n"
        "  • ${JOB_DIR}, env vars, and ~ are expanded; output root defaults to 'results/'.\n"
        "    Override with $PERMUTER_OUTPUT_ROOT or --out."
    ),
)
console = Console()
rich_tb(show_locals=False)


@app.callback()
def _root(verbose: int = typer.Option(0, "--verbose", "-v", count=True)):
    """
    Global flags: -v for more logs (repeatable).
    """
    level = (
        logging.WARNING
        if verbose == 0
        else (logging.INFO if verbose == 1 else logging.DEBUG)
    )
    logging.basicConfig(level=level, format="%(message)s")


@app.command(
    "run",
    help=(
        "Generate variants for a single reference using the configured protocol.\n"
        "Writes one dataset directory per reference with records.parquet, REF.fa and plots/."
    ),
)
def run(
    job: str = typer.Option(..., "--job", "-j", help="Job YAML path or PRESET name."),
    ref: str = typer.Option(
        None,
        "--ref",
        help="Reference name (row in refs CSV). Required if CSV has multiple rows.",
    ),
    out: Path = typer.Option(
        None, "--out", "-o", help="Output root directory (default: job.output.dir)"
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Allow overwriting an existing dataset (records.parquet).",
    ),
):
    run_cmd.run(job=job, ref=ref, out=out, overwrite=overwrite)


@app.command(
    "evaluate",
    help=(
        "Append metric columns to records.parquet using one or more evaluators. "
        "Use --with id:evaluator[:metric] or --job to read evaluate.metrics[]."
    ),
)
def evaluate(
    data: Path = typer.Option(
        None,
        "--data",
        "-d",
        help="Path to records.parquet OR dataset directory (optional if --job/--ref given)",
    ),
    with_spec: List[str] = typer.Option(
        None,
        "--with",
        help="Repeatable: id:evaluator[:metric] (e.g., llr:evo2_llr:log_likelihood_ratio)",
    ),
    metric: List[str] = typer.Option(
        None, "--metric", help="Convenience: metric ids scored by placeholder evaluator"
    ),
    job: str = typer.Option(
        None,
        "--job",
        "-j",
        help="Job YAML path or PRESET name (used if --data omitted)",
    ),
    ref: str = typer.Option(
        None, "--ref", help="Reference name from refs CSV (used if --data omitted)"
    ),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Override output root when deriving dataset from --job/--ref",
    ),
):
    eval_cmd.evaluate(
        data=data,
        metric_ids=list(metric or []),
        with_spec=list(with_spec or []),
        job=job,
        ref=ref,
        out=out,
    )


@app.command(
    "plot",
    help=(
        "Generate plots (PNG) from a dataset. Use --metric-id to choose which metric to visualize."
    ),
)
def plot(
    data: Path = typer.Option(
        ..., "--data", "-d", help="Path to records.parquet OR dataset directory"
    ),
    which: List[str] = typer.Option(
        ["position_scatter_and_heatmap"], "--which", help="Plots to generate"
    ),
    metric_id: str = typer.Option(
        None,
        "--metric-id",
        help="Metric id to plot (defaults to single metric or objective)",
    ),
):
    plot_cmd.plot(data=data, which=which, metric_id=metric_id)


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
):
    validate_cmd.validate(data=data, strict=strict)


@app.command(
    "inspect",
    help="Print a small summary table and the head of the dataset for quick inspection.",
)
def inspect(
    data: Path = typer.Option(..., "--data", "-d", exists=True, readable=True),
    head: int = typer.Option(5, "--head", "-n"),
):
    inspect_cmd.inspect_(data=data, head=head)


def main() -> int:
    try:
        app()
        return 0
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
