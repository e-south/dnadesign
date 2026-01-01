"""Typer CLI entrypoint for cruncher."""

from __future__ import annotations

import typer

from dnadesign.cruncher.cli.commands.analyze import app as analyze_app
from dnadesign.cruncher.cli.commands.cache import app as cache_app
from dnadesign.cruncher.cli.commands.catalog import app as catalog_app
from dnadesign.cruncher.cli.commands.config import app as config_app
from dnadesign.cruncher.cli.commands.fetch import app as fetch_app
from dnadesign.cruncher.cli.commands.lock import app as lock_app
from dnadesign.cruncher.cli.commands.optimizers import app as optimizers_app
from dnadesign.cruncher.cli.commands.parse import app as parse_app
from dnadesign.cruncher.cli.commands.report import app as report_app
from dnadesign.cruncher.cli.commands.runs import app as runs_app
from dnadesign.cruncher.cli.commands.sample import app as sample_app
from dnadesign.cruncher.cli.commands.sources import app as sources_app
from dnadesign.cruncher.cli.commands.targets import app as targets_app
from dnadesign.cruncher.utils.logging import configure_logging

app = typer.Typer(no_args_is_help=True, help="Design short DNA sequences that score highly across TF motifs.")
app.info.epilog = (
    "Command summary:\n"
    "  fetch      Fetch motifs or binding sites into the local cache.\n"
    "  lock       Resolve TF names to exact cached motif IDs.\n"
    "  parse      Validate cached motifs and render PWM logos.\n"
    "  sample     Run MCMC optimization to design sequences.\n"
    "  analyze    Generate diagnostics and plots from runs.\n"
    "  report     Summarize a completed run.\n"
    "  catalog    Inspect cached motifs and sites.\n"
    "  targets    Check TF target readiness.\n"
    "  runs       List, inspect, or watch run artifacts.\n"
)


@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        envvar="CRUNCHER_LOG_LEVEL",
        help="Logging level (e.g., DEBUG, INFO, WARNING).",
    ),
) -> None:
    """Design short DNA sequences that score highly across multiple TF motifs."""
    configure_logging(log_level)


app.add_typer(
    parse_app,
    name="parse",
    help="Validate cached motifs and render PWM logos.",
    short_help="Validate motifs + render logos.",
)
app.add_typer(
    sample_app,
    name="sample",
    help="Run MCMC optimization to design high-scoring sequences.",
    short_help="Run MCMC sequence optimization.",
)
app.add_typer(
    analyze_app,
    name="analyze",
    help="Generate diagnostics and plots from previous runs.",
    short_help="Plot diagnostics for a run.",
)
app.add_typer(
    report_app,
    name="report",
    help="Summarize a completed sample run into report.json/md.",
    short_help="Summarize a run.",
)
app.add_typer(
    fetch_app,
    name="fetch",
    help="Fetch motifs or binding sites into the local cache.",
    short_help="Fetch motifs/sites into cache.",
)
app.add_typer(
    catalog_app,
    name="catalog",
    help="Inspect what motifs and sites are cached locally.",
    short_help="Inspect cached motifs/sites.",
)
app.add_typer(
    lock_app,
    name="lock",
    help="Resolve TF names to exact cached motif IDs.",
    short_help="Resolve TFs to lockfile.",
)
app.add_typer(
    cache_app,
    name="cache",
    help="Inspect or verify the local cache integrity.",
    short_help="Inspect cache integrity.",
)
app.add_typer(
    config_app,
    name="config",
    help="Summarize effective configuration settings.",
    short_help="Summarize config.",
)
app.add_typer(
    sources_app,
    name="sources",
    help="List or inspect ingestion sources and capabilities.",
    short_help="List ingestion sources.",
)
app.add_typer(
    targets_app,
    name="targets",
    help="Check TF target readiness and catalog candidates.",
    short_help="Check TF target readiness.",
)
app.add_typer(
    optimizers_app,
    name="optimizers",
    help="List available optimizer kernels.",
    short_help="List optimizer kernels.",
)
app.add_typer(
    runs_app,
    name="runs",
    help="List, inspect, or watch past run artifacts.",
    short_help="Inspect previous runs.",
)

if __name__ == "__main__":
    app()
