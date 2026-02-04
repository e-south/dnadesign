"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/app.py

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
        "  • --job accepts a path or PRESET NAME (search: $PERMUTER_JOBS, CWD/jobs, repo, package jobs/).\n"
        "  • --data accepts a dataset directory OR a records.parquet file.\n"
        "  • ${JOB_DIR}, env vars, and ~ are expanded. Output root defaults to the job's configured 'results/' path.\n"
        "    No silent fallbacks: if unwritable, use $PERMUTER_OUTPUT_ROOT or --out."
    ),
)
console = Console()
# richer tracebacks with locals by default; no suppression
rich_tb(show_locals=True)


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
    job: str = typer.Option(..., "--job", "-j", help="Job YAML path or PRESET name."),
    ref: str = typer.Option(
        None,
        "--ref",
        help="Reference name (row in refs CSV). Required if CSV has multiple rows.",
    ),
    out: Path = typer.Option(None, "--out", "-o", help="Output root directory (default: job.output.dir)"),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Replace existing dataset (records.parquet) if present.",
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
    metric: List[str] = typer.Option(None, "--metric", help="Convenience: metric ids scored by placeholder evaluator"),
    job: str = typer.Option(
        None,
        "--job",
        "-j",
        help="Job YAML path or PRESET name (used if --data omitted)",
    ),
    ref: str = typer.Option(None, "--ref", help="Reference name from refs CSV (used if --data omitted)"),
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
        "Generate plots (PNG) for a dataset. "
        "Use --metric-id (or job.plot.metric_id) to choose a metric when multiple exist. "
        "Repeat --which to draw multiple plots."
    ),
)
def plot(
    data: Path = typer.Option(None, "--data", "-d", help="Path to records.parquet OR dataset directory"),
    job: str = typer.Option(
        None,
        "--job",
        "-j",
        help="Job YAML path or PRESET name (used if --data omitted)",
    ),
    ref: str = typer.Option(None, "--ref", help="Reference name from refs CSV (used with --job)"),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Override output root when deriving dataset from --job/--ref",
    ),
    which: List[str] = typer.Option(
        None,
        "--which",
        help="Plot id to generate (repeat for multiple). "
        "Allowed: position_scatter_and_heatmap, ranked_variants, synergy_scatter, "
        "metric_by_mutation_count, aa_category_effects, hairpin_length_vs_metric, window_score_mass",  # noqa
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
):
    plot_cmd.plot(
        data=data,
        job=job,
        ref=ref,
        out=out,
        which=which,
        metric_id=metric_id,
        width=width,
        height=height,
        font_scale=font_scale,
        emit_summaries=emit_summaries,
    )


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
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
