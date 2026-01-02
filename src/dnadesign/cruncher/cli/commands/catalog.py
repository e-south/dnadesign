"""Catalog command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.catalog_service import get_entry, list_catalog, search_catalog

app = typer.Typer(no_args_is_help=True, help="Query or inspect cached motifs and binding sites.")
console = Console()


@app.command("list", help="List cached motifs and site sets.")
def list_entries(
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    tf: Optional[str] = typer.Option(None, "--tf", help="Filter by TF name."),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source adapter."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Filter by organism name or strain."),
    include_synonyms: bool = typer.Option(
        False,
        "--include-synonyms",
        help="Match TF synonyms in addition to tf_name.",
    ),
) -> None:
    cfg = load_config(config)
    catalog_root = config.parent / cfg.motif_store.catalog_root
    entries = list_catalog(
        catalog_root,
        tf_name=tf,
        source=source,
        organism=organism,
        include_synonyms=include_synonyms,
    )
    if not entries:
        console.print("No catalog entries found.")
        console.print("Hint: run cruncher fetch motifs --tf <name> <config> to populate the cache.")
        raise typer.Exit(code=1)
    table = Table(title="Catalog", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Organism")
    table.add_column("Matrix")
    table.add_column("Sites (seq/total)")
    table.add_column("Site kind")
    table.add_column("Dataset")
    table.add_column("Method")
    table.add_column("Mean len")
    table.add_column("Updated")
    for entry in entries:
        organism_label = "-"
        if entry.organism:
            organism_label = entry.organism.get("name") or entry.organism.get("strain") or "-"
        matrix_flag = "yes" if entry.has_matrix else "no"
        if entry.has_matrix and entry.matrix_source:
            matrix_flag = f"{matrix_flag} ({entry.matrix_source})"
        sites_flag = f"{entry.site_count}/{entry.site_total}" if entry.has_sites else "no"
        mean_len = "-"
        if entry.site_length_mean is not None:
            mean_len = f"{entry.site_length_mean:.1f}"
        table.add_row(
            entry.tf_name,
            entry.source,
            entry.motif_id,
            organism_label,
            matrix_flag,
            sites_flag,
            entry.site_kind or "-",
            entry.dataset_id or "-",
            entry.dataset_method or "-",
            mean_len,
            entry.updated_at.split("T")[0],
        )
    console.print(table)


@app.command("search", help="Search cached motifs by name or motif ID.")
def search_entries(
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    query: str = typer.Argument(..., help="Search query (TF name, motif ID, or regex).", metavar="QUERY"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source adapter."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Filter by organism name or strain."),
    regex: bool = typer.Option(False, "--regex", help="Treat query as a regular expression."),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", help="Enable case-sensitive matching."),
    fuzzy: bool = typer.Option(False, "--fuzzy", help="Use Levenshtein ratio to rank approximate matches."),
    min_score: float = typer.Option(0.6, "--min-score", help="Minimum fuzzy score (0-1)."),
    limit: Optional[int] = typer.Option(25, "--limit", help="Limit number of returned entries."),
) -> None:
    cfg = load_config(config)
    if fuzzy and regex:
        raise typer.BadParameter(
            "--fuzzy and --regex are mutually exclusive. Hint: use --fuzzy for approximate matches."
        )
    if not (0.0 <= min_score <= 1.0):
        raise typer.BadParameter("--min-score must be between 0 and 1. Hint: try 0.6.")
    catalog_root = config.parent / cfg.motif_store.catalog_root
    entries = search_catalog(
        catalog_root,
        query=query,
        source=source,
        organism=organism,
        regex=regex,
        case_sensitive=case_sensitive,
        fuzzy=fuzzy,
        min_score=min_score,
        limit=limit,
    )
    if not entries:
        console.print(f"No catalog matches for '{query}'.")
        console.print("Hint: run cruncher catalog list <config> to inspect cached entries.")
        raise typer.Exit(code=1)
    table = Table(title=f"Catalog search: {query}", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Organism")
    table.add_column("Matrix")
    table.add_column("Sites (seq/total)")
    table.add_column("Site kind")
    table.add_column("Dataset")
    table.add_column("Method")
    table.add_column("Mean len")
    for entry in entries:
        organism_label = "-"
        if entry.organism:
            organism_label = entry.organism.get("name") or entry.organism.get("strain") or "-"
        matrix_flag = "yes" if entry.has_matrix else "no"
        if entry.has_matrix and entry.matrix_source:
            matrix_flag = f"{matrix_flag} ({entry.matrix_source})"
        sites_flag = f"{entry.site_count}/{entry.site_total}" if entry.has_sites else "no"
        mean_len = "-"
        if entry.site_length_mean is not None:
            mean_len = f"{entry.site_length_mean:.1f}"
        table.add_row(
            entry.tf_name,
            entry.source,
            entry.motif_id,
            organism_label,
            matrix_flag,
            sites_flag,
            entry.site_kind or "-",
            entry.dataset_id or "-",
            entry.dataset_method or "-",
            mean_len,
        )
    console.print(table)


@app.command("resolve", help="Resolve a TF name to cached motif candidates.")
def resolve_tf(
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    tf: str = typer.Argument(..., help="TF name to resolve in the catalog.", metavar="TF"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source adapter."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Filter by organism name or strain."),
    include_synonyms: bool = typer.Option(True, "--include-synonyms", help="Include TF synonyms in resolution."),
) -> None:
    cfg = load_config(config)
    catalog_root = config.parent / cfg.motif_store.catalog_root
    entries = list_catalog(
        catalog_root,
        tf_name=tf,
        source=source,
        organism=organism,
        include_synonyms=include_synonyms,
    )
    if not entries:
        console.print(f"No cached entries for TF '{tf}'.")
        console.print("Hint: run cruncher fetch motifs --tf <name> <config> to populate the cache.")
        raise typer.Exit(code=1)
    table = Table(title=f"TF resolve: {tf}", header_style="bold")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Organism")
    table.add_column("Matrix")
    table.add_column("Sites (seq/total)")
    table.add_column("Site kind")
    table.add_column("Dataset")
    table.add_column("Method")
    table.add_column("Mean len")
    for entry in entries:
        organism_label = "-"
        if entry.organism:
            organism_label = entry.organism.get("name") or entry.organism.get("strain") or "-"
        matrix_flag = "yes" if entry.has_matrix else "no"
        if entry.has_matrix and entry.matrix_source:
            matrix_flag = f"{matrix_flag} ({entry.matrix_source})"
        sites_flag = f"{entry.site_count}/{entry.site_total}" if entry.has_sites else "no"
        mean_len = "-"
        if entry.site_length_mean is not None:
            mean_len = f"{entry.site_length_mean:.1f}"
        table.add_row(
            entry.source,
            entry.motif_id,
            organism_label,
            matrix_flag,
            sites_flag,
            entry.site_kind or "-",
            entry.dataset_id or "-",
            entry.dataset_method or "-",
            mean_len,
        )
    console.print(table)


@app.command("show", help="Show metadata for a cached motif reference.")
def show(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (required).",
        metavar="CONFIG",
    ),
    ref: str | None = typer.Argument(
        None,
        help="Catalog reference (<source>:<motif_id>).",
        metavar="REF",
    ),
) -> None:
    if config is None or ref is None:
        console.print("Missing CONFIG/REF. Example: cruncher catalog show <config> regulondb:RBM000123")
        raise typer.Exit(code=1)
    cfg = load_config(config)
    if ":" not in ref:
        raise typer.BadParameter(
            "Expected <source>:<motif_id> reference. Hint: cruncher catalog show <config> regulondb:RBM000123"
        )
    source, motif_id = ref.split(":", 1)
    catalog_root = config.parent / cfg.motif_store.catalog_root
    entry = get_entry(catalog_root, source=source, motif_id=motif_id)
    if entry is None:
        console.print(f"No catalog entry found for {ref}")
        console.print("Hint: run cruncher catalog list <config> to inspect cached entries.")
        raise typer.Exit(code=1)
    console.print(f"source: {entry.source}")
    console.print(f"motif_id: {entry.motif_id}")
    console.print(f"tf_name: {entry.tf_name}")
    console.print(f"organism: {entry.organism or '-'}")
    console.print(f"kind: {entry.kind}")
    console.print(f"matrix_length: {entry.matrix_length}")
    console.print(f"matrix_source: {entry.matrix_source}")
    console.print(f"matrix_semantics: {entry.matrix_semantics}")
    console.print(f"has_matrix: {entry.has_matrix}")
    console.print(f"has_sites: {entry.has_sites}")
    console.print(f"site_count: {entry.site_count}")
    console.print(f"site_total: {entry.site_total}")
    console.print(f"site_kind: {entry.site_kind or '-'}")
    if entry.site_length_mean is not None:
        console.print(
            f"site_length_mean: {entry.site_length_mean:.2f} "
            f"(min={entry.site_length_min}, max={entry.site_length_max}, n={entry.site_length_count})"
        )
    else:
        console.print("site_length_mean: -")
    console.print(f"site_length_source: {entry.site_length_source or '-'}")
    console.print(f"dataset_id: {entry.dataset_id or '-'}")
    console.print(f"dataset_source: {entry.dataset_source or '-'}")
    console.print(f"dataset_method: {entry.dataset_method or '-'}")
    console.print(f"reference_genome: {entry.reference_genome or '-'}")
    console.print(f"updated_at: {entry.updated_at}")
    synonyms = entry.tags.get("synonyms") if entry.tags else None
    console.print(f"synonyms: {synonyms or '-'}")
    motif_path = catalog_root / "normalized" / "motifs" / entry.source / f"{entry.motif_id}.json"
    sites_path = catalog_root / "normalized" / "sites" / entry.source / f"{entry.motif_id}.jsonl"
    console.print(f"motif_path: {motif_path if motif_path.exists() else '-'}")
    console.print(f"sites_path: {sites_path if sites_path.exists() else '-'}")
