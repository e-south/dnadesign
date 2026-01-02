"""Fetch command."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.ingest.http_client import HttpRetryPolicy
from dnadesign.cruncher.ingest.models import DatasetQuery, MotifQuery
from dnadesign.cruncher.ingest.registry import default_registry
from dnadesign.cruncher.ingest.sequence_provider import (
    FastaSequenceProvider,
    NCBISequenceProvider,
    SequenceProvider,
)
from dnadesign.cruncher.services.fetch_service import fetch_motifs, fetch_sites, hydrate_sites
from dnadesign.cruncher.store.catalog_index import CatalogIndex

app = typer.Typer(no_args_is_help=True, help="Fetch motifs or binding sites from sources into the cache.")
logger = logging.getLogger(__name__)
console = Console()


def _unique_keys(paths: Iterable[Path]) -> List[Tuple[str, str]]:
    keys = {(path.parent.name, path.stem) for path in paths}
    return sorted(keys, key=lambda pair: (pair[0], pair[1]))


def _render_motif_summary(catalog_root: Path, paths: Iterable[Path]) -> None:
    catalog = CatalogIndex.load(catalog_root)
    table = Table(title="Fetched motifs", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Length")
    table.add_column("Matrix")
    table.add_column("Updated")
    for source, motif_id in _unique_keys(paths):
        entry = catalog.entries.get(f"{source}:{motif_id}")
        if entry is None:
            table.add_row("-", source, motif_id, "-", "-", "-")
            continue
        matrix_flag = "yes" if entry.has_matrix else "no"
        if entry.has_matrix and entry.matrix_source:
            matrix_flag = f"{matrix_flag} ({entry.matrix_source})"
        updated = entry.updated_at.split("T")[0] if entry.updated_at else "-"
        table.add_row(
            entry.tf_name,
            entry.source,
            entry.motif_id,
            str(entry.matrix_length or "-"),
            matrix_flag,
            updated,
        )
    console.print(table)


def _render_sites_summary(catalog_root: Path, paths: Iterable[Path]) -> None:
    catalog = CatalogIndex.load(catalog_root)
    table = Table(title="Fetched binding-site sets", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Kind")
    table.add_column("Dataset")
    table.add_column("Method")
    table.add_column("Sites")
    table.add_column("Total")
    table.add_column("Mean len")
    table.add_column("Updated")
    for source, motif_id in _unique_keys(paths):
        entry = catalog.entries.get(f"{source}:{motif_id}")
        if entry is None:
            table.add_row("-", source, motif_id, "-", "-", "-", "-", "-")
            continue
        updated = entry.updated_at.split("T")[0] if entry.updated_at else "-"
        mean_len = "-"
        if entry.site_length_mean is not None:
            mean_len = f"{entry.site_length_mean:.1f}"
        table.add_row(
            entry.tf_name,
            entry.source,
            entry.motif_id,
            entry.site_kind or "-",
            entry.dataset_id or "-",
            entry.dataset_method or "-",
            str(entry.site_count),
            str(entry.site_total),
            mean_len,
            updated,
        )
    console.print(table)


@app.command("motifs", help="Fetch motif matrices into the local cache.")
def motifs(
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    tf: List[str] = typer.Option([], "--tf", help="TF name to fetch (repeatable)."),
    motif_id: List[str] = typer.Option([], "--motif-id", help="Motif ID to fetch (repeatable)."),
    source: str = typer.Option("regulondb", "--source", help="Source adapter to query."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List matching motifs without caching."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit remote matches (dry-run only)."),
    all_matches: bool = typer.Option(False, "--all", help="Fetch all matching motifs for each TF name."),
    update: bool = typer.Option(False, "--update", help="Refresh cached motifs even if present."),
    offline: bool = typer.Option(False, "--offline", help="Do not use network; only verify cached motifs."),
    summary: bool = typer.Option(True, "--summary/--no-summary", help="Show summary table of fetched motifs."),
    paths: bool = typer.Option(False, "--paths", help="Print raw cache paths (for scripting)."),
) -> None:
    cfg = load_config(config)
    if not tf and not motif_id:
        raise typer.BadParameter(
            "Provide at least one --tf or --motif-id. Hint: cruncher fetch motifs --tf lexA <config>"
        )
    if offline and update:
        raise typer.BadParameter(
            "--offline and --update are mutually exclusive. Hint: use --offline to verify cache or --update to refresh."
        )
    if dry_run and offline:
        raise typer.BadParameter("--dry-run cannot be combined with --offline.")
    if dry_run and update:
        raise typer.BadParameter("--dry-run does not write cache; remove --update.")
    try:
        registry = default_registry()
        adapter = registry.create(source, cfg.ingest)
        catalog_root = config.parent / cfg.motif_store.catalog_root
        if dry_run:
            if not tf:
                raise typer.BadParameter("--dry-run requires at least one --tf query.")
            rows: list[tuple[str, str, str, str]] = []
            for name in tf:
                results = adapter.list_motifs(MotifQuery(tf_name=name, limit=limit))
                if not results:
                    raise ValueError(f"No motifs found for {name}")
                for rec in results:
                    organism = "-"
                    if rec.organism is not None:
                        organism = rec.organism.name or rec.organism.strain or "-"
                    rows.append((name, rec.source, rec.motif_id, organism))
            table = Table(title="Remote motif matches", header_style="bold")
            table.add_column("Query")
            table.add_column("Source")
            table.add_column("Motif ID")
            table.add_column("Organism")
            for row in rows:
                table.add_row(*row)
            console.print(table)
            return
        logger.info("Fetching motifs from %s for TFs=%s motif_ids=%s", source, tf, motif_id)
        written = fetch_motifs(
            adapter,
            catalog_root,
            names=tf,
            motif_ids=motif_id,
            fetch_all=all_matches,
            update=update,
            offline=offline,
        )
    except (ValueError, FileNotFoundError, HTTPError, URLError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo("Hint: run cruncher fetch motifs --help for examples.", err=True)
        raise typer.Exit(code=1)
    if not written:
        console.print("No new motifs cached (all matches already present). Use --update to refresh.")
    if summary and written:
        _render_motif_summary(catalog_root, written)
    if paths or not summary:
        for path in written:
            typer.echo(str(path))


@app.command("sites", help="Fetch binding-site sequences into the local cache.")
def sites(
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    tf: List[str] = typer.Option([], "--tf", help="TF name to fetch (repeatable)."),
    motif_id: List[str] = typer.Option([], "--motif-id", help="Motif ID to fetch (repeatable)."),
    source: str = typer.Option("regulondb", "--source", help="Source adapter to query."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optional limit on sites per TF."),
    dataset_id: Optional[str] = typer.Option(
        None,
        "--dataset-id",
        help="Limit HT site retrieval to a specific dataset ID (enables HT for this request).",
    ),
    genome_fasta: Optional[Path] = typer.Option(
        None,
        "--genome-fasta",
        help="Reference genome FASTA to hydrate coordinate-only sites into sequences (overrides ingest.genome_source).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List matching HT datasets without caching."),
    hydrate: bool = typer.Option(
        False,
        "--hydrate",
        help="Hydrate missing sequences in cached sites without refetching.",
    ),
    update: bool = typer.Option(False, "--update", help="Refresh cached sites even if present."),
    offline: bool = typer.Option(False, "--offline", help="Do not use network; only verify cached sites."),
    summary: bool = typer.Option(True, "--summary/--no-summary", help="Show summary table of fetched sites."),
    paths: bool = typer.Option(False, "--paths", help="Print raw cache paths (for scripting)."),
) -> None:
    cfg = load_config(config)
    if not tf and not motif_id:
        raise typer.BadParameter(
            "Provide at least one --tf or --motif-id. Hint: cruncher fetch sites --tf lexA <config>"
        )
    if offline and update:
        raise typer.BadParameter(
            "--offline and --update are mutually exclusive. Hint: use --offline to verify cache or --update to refresh."
        )
    if dry_run and update:
        raise typer.BadParameter("--dry-run does not write cache; remove --update.")
    if dry_run and offline:
        raise typer.BadParameter("--dry-run cannot be combined with --offline.")
    if hydrate and update:
        raise typer.BadParameter("--hydrate cannot be combined with --update.")
    if hydrate and dry_run:
        raise typer.BadParameter("--hydrate cannot be combined with --dry-run.")
    provider: Optional[SequenceProvider] = None
    try:
        registry = default_registry()
        ingest_cfg = cfg.ingest
        if dataset_id and source == "regulondb" and not cfg.ingest.regulondb.ht_sites:
            ingest_cfg = cfg.ingest.model_copy(deep=True)
            ingest_cfg.regulondb.ht_sites = True
            console.print("Note: enabling HT dataset access for this request (--dataset-id).")
        adapter = registry.create(source, ingest_cfg)
        catalog_root = config.parent / cfg.motif_store.catalog_root
        if dry_run:
            if not tf:
                raise typer.BadParameter("--dry-run requires --tf to resolve HT datasets.")
            if not hasattr(adapter, "list_datasets"):
                raise typer.BadParameter(f"Source '{source}' does not support dataset discovery.")
            rows: list[tuple[str, str, str, str, str]] = []
            seen: set[tuple[str, str]] = set()
            for name in tf:
                datasets = adapter.list_datasets(DatasetQuery(tf_name=name))
                if dataset_id:
                    datasets = [ds for ds in datasets if ds.dataset_id == dataset_id]
                if not datasets:
                    raise ValueError(f"No HT datasets found for TF {name}")
                for ds in datasets:
                    key = (name, ds.dataset_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    method = ds.method or "-"
                    genome = ds.reference_genome or "-"
                    rows.append((name, ds.dataset_id, ds.dataset_source or "-", method, genome))
            table = Table(title="HT datasets", header_style="bold")
            table.add_column("TF")
            table.add_column("Dataset ID")
            table.add_column("Source")
            table.add_column("Method")
            table.add_column("Genome")
            for row in rows:
                table.add_row(*row)
            console.print(table)
            return
        genome_path = genome_fasta or cfg.ingest.genome_fasta
        if genome_path is not None:
            if not genome_path.is_absolute():
                genome_path = (config.parent / genome_path).resolve()
            provider = FastaSequenceProvider(
                genome_path,
                assembly_id=cfg.ingest.genome_assembly,
                contig_aliases=cfg.ingest.contig_aliases,
            )
        elif cfg.ingest.genome_source == "fasta":
            raise typer.BadParameter("genome_source=fasta requires ingest.genome_fasta or --genome-fasta.")
        elif cfg.ingest.genome_source == "ncbi":
            genome_cache = cfg.ingest.genome_cache
            if not genome_cache.is_absolute():
                genome_cache = (config.parent / genome_cache).resolve()
            provider = NCBISequenceProvider(
                cache_root=genome_cache,
                email=cfg.ingest.ncbi_email,
                tool=cfg.ingest.ncbi_tool,
                api_key=cfg.ingest.ncbi_api_key,
                timeout_seconds=cfg.ingest.ncbi_timeout_seconds,
                retry_policy=HttpRetryPolicy(
                    retries=cfg.ingest.http.retries,
                    backoff_seconds=cfg.ingest.http.backoff_seconds,
                    max_backoff_seconds=cfg.ingest.http.max_backoff_seconds,
                    retry_statuses=tuple(cfg.ingest.http.retry_statuses),
                    respect_retry_after=cfg.ingest.http.respect_retry_after,
                ),
                refresh=update,
                offline=offline,
                contig_aliases=cfg.ingest.contig_aliases,
            )
        if hydrate:
            if provider is None:
                raise ValueError("Hydration requires genome_source or --genome-fasta.")
            logger.info("Hydrating cached sites for TFs=%s motif_ids=%s", tf, motif_id)
            written = hydrate_sites(
                catalog_root,
                names=tf,
                motif_ids=motif_id,
                sequence_provider=provider,
            )
        else:
            logger.info("Fetching binding sites from %s for TFs=%s motif_ids=%s", source, tf, motif_id)
            written = fetch_sites(
                adapter,
                catalog_root,
                names=tf,
                motif_ids=motif_id,
                limit=limit,
                dataset_id=dataset_id,
                update=update,
                offline=offline,
                sequence_provider=provider,
            )
    except (ValueError, FileNotFoundError, HTTPError, URLError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo("Hint: run cruncher fetch sites --help for examples.", err=True)
        raise typer.Exit(code=1)
    finally:
        if provider is not None:
            provider.close()
    if not written:
        console.print("No new sites cached (all matches already present). Use --update to refresh.")
    if summary and written:
        _render_sites_summary(catalog_root, written)
    if paths or not summary:
        for path in written:
            typer.echo(str(path))
