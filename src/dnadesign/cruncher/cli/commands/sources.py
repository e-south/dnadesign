"""Sources command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.ingest.models import DatasetQuery
from dnadesign.cruncher.ingest.registry import default_registry

app = typer.Typer(no_args_is_help=True, help="List available ingestion sources and capabilities.")
console = Console()


@app.command("list", help="List registered ingestion sources.")
def list_sources() -> None:
    registry = default_registry()
    table = Table(title="Sources", header_style="bold")
    table.add_column("Source")
    table.add_column("Description")
    for spec in registry.list_sources():
        table.add_row(spec.source_id, spec.description)
    console.print(table)


@app.command("info", help="Show capabilities for a specific source.")
def info(
    source: str = typer.Argument(..., help="Source adapter name.", metavar="SOURCE"),
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
) -> None:
    cfg = load_config(config)
    registry = default_registry()
    try:
        adapter = registry.create(source, cfg.ingest)
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    caps = ", ".join(sorted(adapter.capabilities()))
    console.print(f"{source}: {caps}")


@app.command("datasets", help="List available HT datasets for a source (if supported).")
def datasets(
    source: str = typer.Argument(..., help="Source adapter name.", metavar="SOURCE"),
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    tf: Optional[str] = typer.Option(None, "--tf", help="Filter datasets to a TF name."),
    dataset_source: Optional[str] = typer.Option(None, "--dataset-source", help="Filter by dataset source."),
    dataset_type: Optional[str] = typer.Option(None, "--dataset-type", help="Filter by dataset type/method."),
    limit: int = typer.Option(50, "--limit", help="Limit number of datasets displayed."),
) -> None:
    cfg = load_config(config)
    registry = default_registry()
    try:
        adapter = registry.create(source, cfg.ingest)
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    if not hasattr(adapter, "list_datasets"):
        console.print(f"Source '{source}' does not support dataset discovery.")
        raise typer.Exit(code=1)
    datasets = adapter.list_datasets(DatasetQuery(tf_name=tf, dataset_source=dataset_source, dataset_type=dataset_type))
    if not datasets:
        console.print("No datasets found.")
        raise typer.Exit(code=1)
    table = Table(title=f"{source} datasets", header_style="bold")
    table.add_column("Dataset ID")
    table.add_column("Source")
    table.add_column("Method")
    table.add_column("TFs")
    table.add_column("Genome")
    for ds in datasets[:limit]:
        table.add_row(
            ds.dataset_id,
            ds.dataset_source or "-",
            ds.method or "-",
            ", ".join(ds.tf_names[:4]) + ("…" if len(ds.tf_names) > 4 else ""),
            ds.reference_genome or "-",
        )
    console.print(table)
