"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/cache.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer
from dnadesign.cruncher.app.catalog_service import catalog_stats, verify_cache
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Inspect cache stats or verify cached artifacts.")
console = Console()


@app.command("stats", help="Show counts of cached motifs and site sets.")
def stats(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    catalog_root = resolve_catalog_root(config_path, cfg.motif_store.catalog_root)
    stats = catalog_stats(catalog_root)
    table = Table(title="Cache stats", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("entries", str(stats["entries"]))
    table.add_row("motifs", str(stats["motifs"]))
    table.add_row("site_sets", str(stats["site_sets"]))
    console.print(table)


@app.command("verify", help="Verify cached motif/site files are present on disk.")
def verify(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    catalog_root = resolve_catalog_root(config_path, cfg.motif_store.catalog_root)
    issues = verify_cache(catalog_root)
    if not issues:
        console.print("Cache verification OK.")
        return
    console.print("[red]Cache verification failed:[/red]")
    for issue in issues:
        console.print(f"- {issue}")
    raise typer.Exit(code=1)
