"""Cache command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.catalog_service import catalog_stats, verify_cache

app = typer.Typer(no_args_is_help=True, help="Inspect cache stats or verify cached artifacts.")
console = Console()


@app.command("stats", help="Show counts of cached motifs and site sets.")
def stats(config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG")) -> None:
    cfg = load_config(config)
    catalog_root = config.parent / cfg.motif_store.catalog_root
    stats = catalog_stats(catalog_root)
    table = Table(title="Cache stats", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("entries", str(stats["entries"]))
    table.add_row("motifs", str(stats["motifs"]))
    table.add_row("site_sets", str(stats["site_sets"]))
    console.print(table)


@app.command("verify", help="Verify cached motif/site files are present on disk.")
def verify(config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG")) -> None:
    cfg = load_config(config)
    catalog_root = config.parent / cfg.motif_store.catalog_root
    issues = verify_cache(catalog_root)
    if not issues:
        console.print("Cache verification OK.")
        return
    console.print("[red]Cache verification failed:[/red]")
    for issue in issues:
        console.print(f"- {issue}")
    raise typer.Exit(code=1)
