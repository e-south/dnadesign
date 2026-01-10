"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/sources.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError

import typer
from dnadesign.cruncher.cli.config_resolver import (
    CANDIDATE_CONFIG_FILENAMES,
    WORKSPACE_ENV_VAR,
    ConfigResolutionError,
    parse_config_and_value,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.ingest.models import DatasetQuery
from dnadesign.cruncher.ingest.registry import default_registry
from dnadesign.cruncher.services.source_summary_service import (
    summarize_cache,
    summarize_combined,
    summarize_remote,
)
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="List available ingestion sources and capabilities.")
console = Console()


@app.command("list", help="List registered ingestion sources.")
def list_sources(
    config: Path | None = typer.Argument(
        None,
        help="Optional path to cruncher config.yaml (includes local sources).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    registry = default_registry()
    config_path: Path | None = None
    workspace_selector = os.environ.get(WORKSPACE_ENV_VAR)
    if config_option is not None or config is not None or workspace_selector:
        try:
            config_path = resolve_config_path(config_option or config)
        except ConfigResolutionError as exc:
            console.print(str(exc))
            raise typer.Exit(code=1)
    else:
        cwd = Path.cwd().expanduser().resolve()
        matches = [cwd / name for name in CANDIDATE_CONFIG_FILENAMES if (cwd / name).is_file()]
        if len(matches) == 1:
            config_path = matches[0].resolve()
        elif len(matches) > 1:
            rendered = "\n".join(f"- {path.relative_to(cwd).as_posix()}" for path in matches)
            console.print(
                "Multiple config files found in the current directory:\n"
                f"{rendered}\n"
                "Hint: pass --config PATH to disambiguate."
            )
            raise typer.Exit(code=1)
    if config_path is not None:
        cfg = load_config(config_path)
        registry = default_registry(
            cfg.ingest,
            config_path=config_path,
            extra_parser_modules=cfg.io.parsers.extra_modules,
        )
    table = Table(title="Sources", header_style="bold")
    table.add_column("Source")
    table.add_column("Description")
    for spec in registry.list_sources():
        table.add_row(spec.source_id, spec.description)
    console.print(table)


@app.command("info", help="Show capabilities for a specific source.")
def info(
    args: list[str] = typer.Argument(
        None,
        help="Source name (optionally followed by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    try:
        config_path, source = parse_config_and_value(
            args,
            config_option,
            value_label="SOURCE",
            command_hint="cruncher sources info regulondb",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    registry = default_registry(
        cfg.ingest,
        config_path=config_path,
        extra_parser_modules=cfg.io.parsers.extra_modules,
    )
    try:
        adapter = registry.create(source, cfg.ingest)
    except (ValueError, HTTPError, URLError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    try:
        caps = ", ".join(sorted(adapter.capabilities()))
    except (RuntimeError, HTTPError, URLError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    console.print(f"{source}: {caps}")


@app.command("datasets", help="List available HT datasets for a source (if supported).")
def datasets(
    args: list[str] = typer.Argument(
        None,
        help="Source name (optionally followed by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    tf: Optional[str] = typer.Option(None, "--tf", help="Filter datasets to a TF name."),
    dataset_source: Optional[str] = typer.Option(None, "--dataset-source", help="Filter by dataset source."),
    dataset_type: Optional[str] = typer.Option(None, "--dataset-type", help="Filter by dataset type/method."),
    limit: int = typer.Option(50, "--limit", help="Limit number of datasets displayed."),
) -> None:
    try:
        config_path, source = parse_config_and_value(
            args,
            config_option,
            value_label="SOURCE",
            command_hint="cruncher sources datasets regulondb",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    registry = default_registry(
        cfg.ingest,
        config_path=config_path,
        extra_parser_modules=cfg.io.parsers.extra_modules,
    )
    try:
        adapter = registry.create(source, cfg.ingest)
    except (ValueError, HTTPError, URLError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    if not hasattr(adapter, "list_datasets"):
        console.print(f"Source '{source}' does not support dataset discovery.")
        raise typer.Exit(code=1)
    try:
        datasets = adapter.list_datasets(
            DatasetQuery(tf_name=tf, dataset_source=dataset_source, dataset_type=dataset_type)
        )
    except (RuntimeError, HTTPError, URLError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    if not datasets:
        console.print("No datasets found.")
        raise typer.Exit(code=1)
    seen: set[str] = set()
    unique = []
    for ds in datasets:
        if ds.dataset_id in seen:
            continue
        seen.add(ds.dataset_id)
        unique.append(ds)
    datasets = unique
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


@app.command("summary", help="Summarize local cache and remote inventories for sources.")
def summary(
    config: Path | None = typer.Argument(None, help="Path to cruncher config.yaml.", metavar="CONFIG"),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    source: Optional[str] = typer.Option(None, "--source", help="Limit summary to a single source."),
    scope: str = typer.Option(
        "both",
        "--scope",
        help="Which summaries to include: cache, remote, or both.",
    ),
    limit: Optional[int] = typer.Option(50, "--limit", help="Limit regulators shown in tables."),
    all_rows: bool = typer.Option(False, "--all", help="Show all regulators in tables."),
    remote_limit: Optional[int] = typer.Option(
        None,
        "--remote-limit",
        help="Limit remote regulators fetched (affects counts; required if a source lacks full inventory).",
    ),
    page_size: int = typer.Option(200, "--page-size", help="Remote page size for inventory listing."),
    sort_by: str = typer.Option(
        "sites_total",
        "--sort-by",
        help="Sort regulators table by: tf, motifs, site_sets, sites_seq, sites_total, datasets.",
    ),
    view: str = typer.Option(
        "split",
        "--view",
        help="Table view: split (cache/remote tables) or combined (single inventory table).",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json.",
    ),
    json_out: Optional[Path] = typer.Option(
        None,
        "--json-out",
        help="Write JSON summary to a file (use '-' to emit JSON to stdout).",
    ),
) -> None:
    scope = scope.lower()
    if scope not in {"cache", "remote", "both"}:
        raise typer.BadParameter("--scope must be one of: cache, remote, both.")
    output_format = output_format.lower()
    if output_format not in {"table", "json"}:
        raise typer.BadParameter("--format must be one of: table, json.")
    view = view.lower()
    if view not in {"split", "combined"}:
        raise typer.BadParameter("--view must be one of: split, combined.")
    if sort_by not in {"tf", "motifs", "site_sets", "sites_seq", "sites_total", "datasets"}:
        raise typer.BadParameter("--sort-by must be one of: tf, motifs, site_sets, sites_seq, sites_total, datasets.")
    if all_rows:
        limit = None
    if page_size < 1:
        raise typer.BadParameter("--page-size must be >= 1.")
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    registry = default_registry(
        cfg.ingest,
        config_path=config_path,
        extra_parser_modules=cfg.io.parsers.extra_modules,
    )
    sources = [source] if source else [spec.source_id for spec in registry.list_sources()]
    payload: dict[str, object] = {}
    cache_summary: dict | None = None
    remote_summaries: dict[str, dict] | None = None
    combined_summary: dict | None = None

    if scope in {"cache", "both"}:
        catalog_root = config_path.parent / cfg.motif_store.catalog_root
        cache_summary = summarize_cache(catalog_root, source=source)
        payload["cache"] = cache_summary
        if output_format == "table" and view == "split":
            _render_cache_tables(cache_summary, source=source)
            cache_regs = _sort_regulators(cache_summary["regulators"], sort_by)
            if cache_regs:
                _render_regulators_table(
                    _title_with_scope("Cache regulators", source=source),
                    cache_regs,
                    limit=limit,
                    include_sites=True,
                )
            else:
                _render_empty_regulators("cache", source)

    if scope in {"remote", "both"}:
        remote_summaries = {}
        for source_id in sources:
            try:
                adapter = registry.create(source_id, cfg.ingest)
            except (ValueError, HTTPError, URLError) as exc:
                console.print(f"Error: {exc}")
                raise typer.Exit(code=1)
            if remote_limit is None and hasattr(adapter, "capabilities"):
                try:
                    caps = adapter.capabilities()
                except (RuntimeError, HTTPError, URLError) as exc:
                    console.print(f"Error: {exc}")
                    raise typer.Exit(code=1)
                supports_iter = "motifs:iter" in caps and hasattr(adapter, "iter_motifs")
                if not supports_iter:
                    console.print(
                        f"Error: Source '{source_id}' does not support full inventory; "
                        "pass --remote-limit (for a partial summary) or use --scope cache."
                    )
                    raise typer.Exit(code=1)
            try:
                remote_summaries[source_id] = summarize_remote(
                    adapter,
                    limit=remote_limit,
                    page_size=page_size,
                    include_datasets=True,
                )
            except (ValueError, RuntimeError, HTTPError, URLError) as exc:
                console.print(f"Error: {exc}")
                raise typer.Exit(code=1)
        payload["remote"] = {"sources": remote_summaries}
        if output_format == "table" and view == "split":
            _render_remote_tables(remote_summaries, remote_limit=remote_limit)
            for source_id, summary_payload in remote_summaries.items():
                remote_sort_by = _resolve_sort_key(sort_by, include_sites=False)
                remote_regs = _sort_regulators(summary_payload["regulators"], remote_sort_by)
                if remote_regs:
                    _render_regulators_table(
                        _title_with_scope(f"Remote regulators: {source_id}", limit=remote_limit),
                        remote_regs,
                        limit=limit,
                        include_sites=False,
                    )
                else:
                    _render_empty_regulators("remote", source_id)

    if output_format == "table" and view == "combined":
        combined_summary = summarize_combined(cache_summary=cache_summary, remote_summaries=remote_summaries)
        _render_combined_overview(combined_summary, source=source, remote_limit=remote_limit)
        combined_regs = _sort_combined_regulators(combined_summary["regulators"], sort_by)
        if combined_regs:
            _render_combined_regulators_table(
                combined_regs,
                limit=limit,
                title=_title_with_scope("Combined regulators (cache + remote)", source=source, limit=remote_limit),
            )
        else:
            _render_empty_regulators("combined", source)

    if output_format == "json" or json_out is not None:
        if combined_summary is None and (view == "combined" or scope in {"both", "cache", "remote"}):
            combined_summary = summarize_combined(cache_summary=cache_summary, remote_summaries=remote_summaries)
        if combined_summary is not None:
            payload["combined"] = combined_summary

    if output_format == "json" or json_out is not None:
        raw = json.dumps(payload, indent=2)
        if json_out is None or str(json_out) == "-":
            if output_format != "json":
                raise typer.BadParameter("--json-out '-' requires --format json to avoid mixed output.")
            console.print(raw, markup=False)
        else:
            json_out.write_text(raw)


def _render_cache_tables(summary: dict, *, source: Optional[str] = None) -> None:
    totals = summary["totals"]
    sources = summary["sources"]
    overview = Table(title=_title_with_scope("Cache overview", source=source), header_style="bold")
    overview.add_column("Metric")
    overview.add_column("Value")
    overview.add_row("entries", _fmt_int(totals["entries"]))
    overview.add_row("sources", _fmt_int(len(sources)))
    overview.add_row("TFs", _fmt_int(totals["tfs"]))
    overview.add_row("motifs", _fmt_int(totals["motifs"]))
    overview.add_row("site sets", _fmt_int(totals["site_sets"]))
    overview.add_row("sites (seq/total)", _fmt_sites(totals["sites_seq"], totals["sites_total"]))
    overview.add_row("datasets", _fmt_int(totals["datasets"]))
    console.print(overview)
    if sources:
        table = Table(title=_title_with_scope("Cache by source", source=source), header_style="bold")
        table.add_column("Source")
        table.add_column("TFs", justify="right")
        table.add_column("Motifs", justify="right")
        table.add_column("Site sets", justify="right")
        table.add_column("Sites (seq/total)")
        table.add_column("Datasets", justify="right")
        for source_id, stats in sorted(sources.items()):
            table.add_row(
                source_id,
                _fmt_int(stats["tfs"]),
                _fmt_int(stats["motifs"]),
                _fmt_int(stats["site_sets"]),
                _fmt_sites(stats["sites_seq"], stats["sites_total"]),
                _fmt_int(stats["datasets"]),
            )
        console.print(table)


def _render_remote_tables(summaries: dict[str, dict], *, remote_limit: Optional[int] = None) -> None:
    if not summaries:
        return
    table = Table(
        title=_title_with_scope("Remote inventory by source", limit=remote_limit),
        header_style="bold",
    )
    table.add_column("Source")
    table.add_column("TFs", justify="right")
    table.add_column("Motifs", justify="right")
    table.add_column("Datasets", justify="right")
    for source_id, stats in sorted(summaries.items()):
        totals = stats["totals"]
        table.add_row(
            source_id,
            _fmt_int(totals["tfs"]),
            _fmt_int(totals["motifs"]),
            _fmt_int(totals["datasets"]),
        )
    console.print(table)


def _render_combined_overview(
    summary: dict,
    *,
    source: Optional[str] = None,
    remote_limit: Optional[int] = None,
) -> None:
    totals = summary.get("totals") or {}
    cache = totals.get("cache") or {}
    remote = totals.get("remote") or {}
    table = Table(title=_title_with_scope("Inventory overview", source=source, limit=remote_limit), header_style="bold")
    table.add_column("Metric")
    table.add_column("Cache")
    table.add_column("Remote")
    table.add_row("TFs", _fmt_int(cache.get("tfs")), _fmt_int(remote.get("tfs")))
    table.add_row("Motifs", _fmt_int(cache.get("motifs")), _fmt_int(remote.get("motifs")))
    table.add_row("Site sets", _fmt_int(cache.get("site_sets")), _fmt_int(None))
    table.add_row("Sites (seq/total)", _fmt_sites(cache.get("sites_seq"), cache.get("sites_total")), "-")
    table.add_row("Datasets", _fmt_int(cache.get("datasets")), _fmt_int(remote.get("datasets")))
    console.print(table)


def _render_combined_regulators_table(
    regulators: list[dict],
    *,
    limit: Optional[int],
    title: Optional[str] = None,
) -> None:
    total = len(regulators)
    shown = regulators
    if limit is not None and total > limit:
        shown = regulators[:limit]
        console.print(f"Showing {limit} of {total} regulators. Use --all to show all.")
    table = Table(title=title or "Combined regulators (cache + remote)", header_style="bold")
    table.add_column("TF")
    table.add_column("Cache sources")
    table.add_column("Remote sources")
    table.add_column("Motifs (cache/remote)", justify="right")
    table.add_column("Site sets", justify="right")
    table.add_column("Sites (seq/total)")
    table.add_column("Datasets (cache/remote)", justify="right")
    for reg in shown:
        cache = reg.get("cache") or {}
        remote = reg.get("remote") or {}
        table.add_row(
            reg.get("tf_name") or "-",
            ", ".join(cache.get("sources") or []) or "-",
            ", ".join(remote.get("sources") or []) or "-",
            _fmt_pair(cache.get("motifs"), remote.get("motifs")),
            _fmt_int(cache.get("site_sets")),
            _fmt_sites(cache.get("sites_seq"), cache.get("sites_total")),
            _fmt_pair(cache.get("datasets"), remote.get("datasets")),
        )
    console.print(table)


def _render_regulators_table(
    title: str,
    regulators: list[dict],
    *,
    limit: Optional[int],
    include_sites: bool,
) -> None:
    total = len(regulators)
    shown = regulators
    if limit is not None and total > limit:
        shown = regulators[:limit]
        console.print(f"Showing {limit} of {total} regulators. Use --all to show all.")
    table = Table(title=title, header_style="bold")
    table.add_column("TF")
    table.add_column("Sources")
    table.add_column("Motifs", justify="right")
    if include_sites:
        table.add_column("Site sets", justify="right")
        table.add_column("Sites (seq/total)")
    table.add_column("Datasets", justify="right")
    for reg in shown:
        row = [
            reg["tf_name"],
            ", ".join(reg.get("sources") or []) or "-",
            _fmt_int(reg.get("motifs")),
        ]
        if include_sites:
            row.extend(
                [
                    _fmt_int(reg.get("site_sets")),
                    _fmt_sites(reg.get("sites_seq"), reg.get("sites_total")),
                ]
            )
        row.append(_fmt_int(reg.get("datasets")))
        table.add_row(*row)
    console.print(table)


def _sort_regulators(regulators: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "tf":
        return sorted(regulators, key=lambda reg: reg["tf_name"].lower())
    return sorted(
        regulators,
        key=lambda reg: (-(reg.get(sort_by) or 0), reg["tf_name"].lower()),
    )


def _sort_combined_regulators(regulators: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "tf":
        return sorted(regulators, key=lambda reg: (reg.get("tf_name") or "").lower())

    def metric(reg: dict) -> int:
        cache = reg.get("cache") or {}
        remote = reg.get("remote") or {}
        if sort_by in {"site_sets", "sites_seq", "sites_total"}:
            return int(cache.get(sort_by) or 0)
        if sort_by == "motifs":
            return int(cache.get("motifs") or 0) + int(remote.get("motifs") or 0)
        if sort_by == "datasets":
            return int(cache.get("datasets") or 0) + int(remote.get("datasets") or 0)
        return 0

    return sorted(
        regulators,
        key=lambda reg: (-metric(reg), (reg.get("tf_name") or "").lower()),
    )


def _resolve_sort_key(sort_by: str, *, include_sites: bool) -> str:
    if include_sites:
        return sort_by
    if sort_by in {"site_sets", "sites_seq", "sites_total"}:
        return "motifs"
    return sort_by


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def _fmt_sites(seq: Optional[int], total: Optional[int]) -> str:
    if seq is None and total is None:
        return "-"
    return f"{seq or 0:,}/{total or 0:,}"


def _fmt_pair(left: Optional[int], right: Optional[int]) -> str:
    if left is None and right is None:
        return "-"
    return f"{left or 0:,}/{right or 0:,}"


def _title_with_scope(
    base: str,
    *,
    source: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    suffixes: list[str] = []
    if source:
        suffixes.append(f"source={source}")
    if limit is not None:
        suffixes.append(f"limit={limit}")
    if not suffixes:
        return base
    return f"{base} ({', '.join(suffixes)})"


def _render_empty_regulators(scope: str, source: Optional[str]) -> None:
    label = f"source '{source}'" if source else "the selected scope"
    console.print(f"No regulators found in {scope} inventory for {label}.")
