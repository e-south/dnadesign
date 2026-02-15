"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/app.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from dnadesign.cruncher.cli.commands.analyze import analyze as analyze_cmd
from dnadesign.cruncher.cli.commands.cache import app as cache_app
from dnadesign.cruncher.cli.commands.campaign import app as campaign_app
from dnadesign.cruncher.cli.commands.catalog import app as catalog_app
from dnadesign.cruncher.cli.commands.config import app as config_app
from dnadesign.cruncher.cli.commands.discover import app as discover_app
from dnadesign.cruncher.cli.commands.doctor import doctor as doctor_cmd
from dnadesign.cruncher.cli.commands.export import app as export_app
from dnadesign.cruncher.cli.commands.fetch import app as fetch_app
from dnadesign.cruncher.cli.commands.lock import lock as lock_cmd
from dnadesign.cruncher.cli.commands.notebook import notebook as notebook_cmd
from dnadesign.cruncher.cli.commands.optimizers import app as optimizers_app
from dnadesign.cruncher.cli.commands.parse import parse as parse_cmd
from dnadesign.cruncher.cli.commands.runs import app as runs_app
from dnadesign.cruncher.cli.commands.sample import sample as sample_cmd
from dnadesign.cruncher.cli.commands.sources import app as sources_app
from dnadesign.cruncher.cli.commands.status import status as status_cmd
from dnadesign.cruncher.cli.commands.targets import app as targets_app
from dnadesign.cruncher.cli.commands.workspaces import app as workspaces_app
from dnadesign.cruncher.cli.config_resolver import CONFIG_ENV_VAR, WORKSPACE_ENV_VAR
from dnadesign.cruncher.utils.logging import configure_logging

app = typer.Typer(
    no_args_is_help=True,
    help="Design short DNA sequences that score highly across TF motifs.",
)
app.info.epilog = "Tip: run `cruncher <command> --help` for examples and details."


@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        envvar="CRUNCHER_LOG_LEVEL",
        help="Logging level (e.g., DEBUG, INFO, WARNING).",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        envvar=CONFIG_ENV_VAR,
        help="Path to cruncher config.yaml (overrides workspace/CWD resolution).",
    ),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        "-w",
        envvar=WORKSPACE_ENV_VAR,
        help="Select a workspace by name, index, or path.",
    ),
) -> None:
    """Design short DNA sequences that score highly across multiple TF motifs."""
    configure_logging(log_level)
    if config:
        os.environ[CONFIG_ENV_VAR] = str(config)
    if workspace:
        os.environ[WORKSPACE_ENV_VAR] = workspace


app.command(
    "parse",
    help="Validate cached motifs and summarize locked PWMs.",
    short_help="Validate locked motifs.",
)(parse_cmd)
app.command(
    "sample",
    help="Run MCMC optimization to design high-scoring sequences.",
    short_help="Run MCMC sequence optimization.",
)(sample_cmd)
app.command(
    "analyze",
    help="Generate diagnostics and plots from previous runs.",
    short_help="Plot diagnostics for a run.",
)(analyze_cmd)
app.add_typer(
    export_app,
    name="export",
    help="Export sequence-centric tables from sample runs.",
    short_help="Export sequence tables.",
)
app.command(
    "notebook",
    help="Generate an optional marimo notebook for analysis.",
    short_help="Generate a marimo notebook.",
)(notebook_cmd)
app.add_typer(
    campaign_app,
    name="campaign",
    help="Generate or summarize regulator campaigns.",
    short_help="Generate campaign configs.",
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
    discover_app,
    name="discover",
    help="Discover motifs from cached binding sites (MEME Suite).",
    short_help="Discover motifs.",
)
app.command(
    "doctor",
    help="Check external dependencies (e.g., MEME Suite).",
    short_help="Check dependencies.",
)(doctor_cmd)
app.command(
    "lock",
    help="Resolve TF names to exact cached motif IDs.",
    short_help="Resolve TFs to lockfile.",
)(lock_cmd)
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
app.command(
    "status",
    help="Show a bird's-eye view of cache, targets, and runs.",
    short_help="Show a status dashboard.",
)(status_cmd)
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
app.add_typer(
    workspaces_app,
    name="workspaces",
    help="List discoverable workspaces.",
    short_help="List workspaces.",
)

if __name__ == "__main__":
    app()
