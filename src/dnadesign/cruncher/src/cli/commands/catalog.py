"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/catalog.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import typer
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    parse_config_and_value,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.campaign_service import select_catalog_entry
from dnadesign.cruncher.services.catalog_service import (
    get_entry,
    list_catalog,
    search_catalog,
)
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.labels import build_run_name
from dnadesign.cruncher.utils.logos import logo_subtitle, site_entries_for_logo
from dnadesign.cruncher.utils.mpl import ensure_mpl_cache
from dnadesign.cruncher.utils.run_layout import logos_dir_for_run, out_root
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Query or inspect cached motifs and binding sites.")
console = Console()
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ResolvedTarget:
    tf_name: str
    ref: MotifRef
    entry: CatalogEntry
    site_entries: list[CatalogEntry]


def _safe_stem(label: str) -> str:
    cleaned = _SAFE_RE.sub("_", label).strip("_")
    return cleaned or "motif"


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


def _resolve_set_tfs(cfg, set_index: int | None) -> list[str]:
    if set_index is None:
        return []
    if set_index < 1 or set_index > len(cfg.regulator_sets):
        raise typer.BadParameter(f"--set must be between 1 and {len(cfg.regulator_sets)} (got {set_index}).")
    return list(cfg.regulator_sets[set_index - 1])


def _parse_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise typer.BadParameter(
            "Expected <source>:<motif_id> reference. Hint: cruncher catalog show regulondb:RDBECOLITFC00214"
        )
    source, motif_id = ref.split(":", 1)
    return source, motif_id


def _ensure_entry_matches_pwm_source(
    entry: CatalogEntry,
    pwm_source: str,
    site_kinds: list[str] | None,
    *,
    tf_name: str,
    ref: str,
) -> None:
    if pwm_source == "matrix":
        if not entry.has_matrix:
            raise ValueError(f"{ref} does not have a cached motif matrix for TF '{tf_name}'.")
        return
    if pwm_source == "sites":
        if not entry.has_sites:
            raise ValueError(f"{ref} does not have cached binding sites for TF '{tf_name}'.")
        if site_kinds is not None and entry.site_kind not in site_kinds:
            raise ValueError(
                f"{ref} site kind '{entry.site_kind}' is not in site_kinds={site_kinds} for TF '{tf_name}'."
            )
        return
    raise ValueError("pwm_source must be 'matrix' or 'sites'")


def _resolve_targets(
    *,
    cfg,
    config_path: Path,
    tfs: Sequence[str],
    refs: Sequence[str],
    set_index: int | None,
    source_filter: str | None,
) -> tuple[list[ResolvedTarget], CatalogIndex]:
    if set_index is not None and (tfs or refs):
        raise typer.BadParameter("--set cannot be combined with --tf or --ref.")
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
    catalog = CatalogIndex.load(catalog_root)
    tf_names = list(tfs)
    if set_index is not None:
        tf_names = _resolve_set_tfs(cfg, set_index)
        if not tf_names and not refs:
            raise typer.BadParameter(f"regulator_sets[{set_index}] is empty.")
    if not tf_names and not refs:
        tf_names = [tf for group in cfg.regulator_sets for tf in group]
    tf_names = _dedupe(tf_names)
    refs = _dedupe(refs)
    if not tf_names and not refs:
        raise typer.BadParameter("No targets resolved. Provide --tf, --ref, or --set.")

    targets: list[ResolvedTarget] = []
    seen_keys: set[str] = set()

    for ref in refs:
        source, motif_id = _parse_ref(ref)
        if source_filter and source_filter != source:
            raise typer.BadParameter(f"--source {source_filter} does not match explicit ref {ref}.")
        entry = catalog.entries.get(f"{source}:{motif_id}")
        if entry is None:
            raise ValueError(f"No catalog entry found for {ref}.")
        _ensure_entry_matches_pwm_source(
            entry,
            cfg.motif_store.pwm_source,
            cfg.motif_store.site_kinds,
            tf_name=entry.tf_name,
            ref=ref,
        )
        site_entries = []
        if cfg.motif_store.pwm_source == "sites":
            site_entries = site_entries_for_logo(
                catalog=catalog,
                entry=entry,
                combine_sites=cfg.motif_store.combine_sites,
                site_kinds=cfg.motif_store.site_kinds,
            )
        key = entry.key
        if key in seen_keys:
            continue
        targets.append(
            ResolvedTarget(
                tf_name=entry.tf_name,
                ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                entry=entry,
                site_entries=site_entries,
            )
        )
        seen_keys.add(key)

    if tf_names:
        catalog_for_select = catalog
        if source_filter:
            filtered = {k: v for k, v in catalog.entries.items() if v.source == source_filter}
            catalog_for_select = CatalogIndex(entries=filtered)
        for tf_name in tf_names:
            if source_filter:
                all_candidates = catalog.list(tf_name=tf_name, include_synonyms=True)
                if not any(candidate.source == source_filter for candidate in all_candidates):
                    raise ValueError(f"No cached entries for '{tf_name}' in source '{source_filter}'.")
            entry = select_catalog_entry(
                catalog=catalog_for_select,
                tf_name=tf_name,
                pwm_source=cfg.motif_store.pwm_source,
                site_kinds=cfg.motif_store.site_kinds,
                combine_sites=cfg.motif_store.combine_sites,
                source_preference=cfg.motif_store.source_preference,
                dataset_preference=cfg.motif_store.dataset_preference,
                dataset_map=cfg.motif_store.dataset_map,
                allow_ambiguous=cfg.motif_store.allow_ambiguous,
            )
            site_entries = []
            if cfg.motif_store.pwm_source == "sites":
                site_entries = site_entries_for_logo(
                    catalog=catalog,
                    entry=entry,
                    combine_sites=cfg.motif_store.combine_sites,
                    site_kinds=cfg.motif_store.site_kinds,
                )
                if not site_entries:
                    raise ValueError(
                        f"No cached site entries available for '{tf_name}' "
                        f"with site_kinds={cfg.motif_store.site_kinds}."
                    )
            key = entry.key
            if key in seen_keys:
                continue
            targets.append(
                ResolvedTarget(
                    tf_name=tf_name,
                    ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                    entry=entry,
                    site_entries=site_entries,
                )
            )
            seen_keys.add(key)

    if not targets:
        raise typer.BadParameter("No catalog entries matched the requested targets.")
    return targets, catalog


def _render_pwm_matrix(table_title: str, pwm_matrix: list[list[float]]) -> Table:
    table = Table(title=table_title, header_style="bold")
    table.add_column("Pos", justify="right")
    table.add_column("A")
    table.add_column("C")
    table.add_column("G")
    table.add_column("T")
    for idx, row in enumerate(pwm_matrix, start=1):
        table.add_row(str(idx), *(f"{val:.3f}" for val in row))
    return table


@app.command("list", help="List cached motifs and site sets.")
def list_entries(
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
    tf: Optional[str] = typer.Option(None, "--tf", help="Filter by TF name."),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source adapter."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Filter by organism name or strain."),
    include_synonyms: bool = typer.Option(
        False,
        "--include-synonyms",
        help="Match TF synonyms in addition to tf_name.",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
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
    args: list[str] = typer.Argument(
        None,
        help="Query (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source adapter."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Filter by organism name or strain."),
    regex: bool = typer.Option(False, "--regex", help="Treat query as a regular expression."),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", help="Enable case-sensitive matching."),
    fuzzy: bool = typer.Option(False, "--fuzzy", help="Use Levenshtein ratio to rank approximate matches."),
    min_score: float = typer.Option(0.6, "--min-score", help="Minimum fuzzy score (0-1)."),
    limit: Optional[int] = typer.Option(25, "--limit", help="Limit number of returned entries."),
) -> None:
    try:
        config_path, query = parse_config_and_value(
            args,
            config_option,
            value_label="QUERY",
            command_hint="cruncher catalog search <query>",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if fuzzy and regex:
        raise typer.BadParameter(
            "--fuzzy and --regex are mutually exclusive. Hint: use --fuzzy for approximate matches."
        )
    if not (0.0 <= min_score <= 1.0):
        raise typer.BadParameter("--min-score must be between 0 and 1. Hint: try 0.6.")
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
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
    args: list[str] = typer.Argument(
        None,
        help="TF name (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source adapter."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Filter by organism name or strain."),
    include_synonyms: bool = typer.Option(True, "--include-synonyms", help="Include TF synonyms in resolution."),
) -> None:
    try:
        config_path, tf = parse_config_and_value(
            args,
            config_option,
            value_label="TF",
            command_hint="cruncher catalog resolve <tf_name>",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
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
    args: list[str] = typer.Argument(
        None,
        help="Catalog reference (<source>:<motif_id>), optionally preceded by CONFIG.",
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
        config_path, ref = parse_config_and_value(
            args,
            config_option,
            value_label="REF",
            command_hint="cruncher catalog show regulondb:RBM000123",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if ":" not in ref:
        raise typer.BadParameter(
            "Expected <source>:<motif_id> reference. Hint: cruncher catalog show regulondb:RBM000123"
        )
    source, motif_id = ref.split(":", 1)
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
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
    console.print(f"motif_path: {render_path(motif_path, base=config_path.parent) if motif_path.exists() else '-'}")
    console.print(f"sites_path: {render_path(sites_path, base=config_path.parent) if sites_path.exists() else '-'}")


@app.command("pwms", help="Summarize or export cached PWMs for selected TFs or motif refs.")
def pwms(
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
    tf: list[str] = typer.Option([], "--tf", help="TF name to include (repeatable)."),
    ref: list[str] = typer.Option([], "--ref", help="Catalog reference (<source>:<motif_id>, repeatable)."),
    set_index: int | None = typer.Option(
        None,
        "--set",
        help="Regulator set index from config (1-based).",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Limit TF resolution to a single source adapter.",
    ),
    matrix: bool = typer.Option(False, "--matrix", help="Print full PWM matrices after the summary."),
    log_odds: bool = typer.Option(False, "--log-odds", help="Also emit log-odds matrices (table or JSON)."),
    output_format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json.",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if output_format not in {"table", "json"}:
        raise typer.BadParameter("--format must be 'table' or 'json'.")
    if log_odds and output_format == "table" and not matrix:
        raise typer.BadParameter("--log-odds requires --matrix for table output.")
    try:
        targets, catalog = _resolve_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source_filter=source,
        )
        store = CatalogMotifStore(
            config_path.parent / cfg.motif_store.catalog_root,
            pwm_source=cfg.motif_store.pwm_source,
            site_kinds=cfg.motif_store.site_kinds,
            combine_sites=cfg.motif_store.combine_sites,
            site_window_lengths=cfg.motif_store.site_window_lengths,
            site_window_center=cfg.motif_store.site_window_center,
            pwm_window_lengths=cfg.motif_store.pwm_window_lengths,
            pwm_window_strategy=cfg.motif_store.pwm_window_strategy,
            min_sites_for_pwm=cfg.motif_store.min_sites_for_pwm,
            allow_low_sites=cfg.motif_store.allow_low_sites,
            pseudocounts=cfg.motif_store.pseudocounts,
        )
        payloads: list[dict[str, object]] = []
        resolved: list[tuple[ResolvedTarget, object]] = []
        table = Table(title="PWM summary", header_style="bold")
        table.add_column("TF")
        table.add_column("Source")
        table.add_column("Motif ID")
        table.add_column("PWM source")
        table.add_column("Length")
        table.add_column("Window")
        table.add_column("Bits")
        table.add_column("n sites")
        table.add_column("Site sets")
        for target in targets:
            pwm = store.get_pwm(target.ref)
            resolved.append((target, pwm))
            info_bits = pwm.information_bits()
            site_sets = "-" if cfg.motif_store.pwm_source == "matrix" else str(len(target.site_entries))
            window = "-"
            if pwm.source_length is not None and pwm.window_start is not None:
                window = f"{pwm.window_start}:{pwm.window_start + pwm.length}/{pwm.source_length}"
            table.add_row(
                target.tf_name,
                target.entry.source,
                target.entry.motif_id,
                cfg.motif_store.pwm_source,
                str(pwm.length),
                window,
                f"{info_bits:.2f}",
                str(pwm.nsites or "-"),
                site_sets,
            )
            record = {
                "tf_name": target.tf_name,
                "source": target.entry.source,
                "motif_id": target.entry.motif_id,
                "pwm_source": cfg.motif_store.pwm_source,
                "length": pwm.length,
                "window_start": pwm.window_start,
                "source_length": pwm.source_length,
                "window_strategy": pwm.window_strategy,
                "window_score": pwm.window_score,
                "info_bits": info_bits,
                "nsites": pwm.nsites,
                "site_sets": len(target.site_entries) if cfg.motif_store.pwm_source == "sites" else None,
            }
            if output_format == "json":
                record["matrix"] = pwm.matrix.tolist()
                if log_odds:
                    record["log_odds"] = pwm.log_odds().tolist()
            payloads.append(record)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch motifs/sites before catalog pwms.")
        raise typer.Exit(code=1)

    if output_format == "json":
        typer.echo(json.dumps(payloads, indent=2))
        return
    console.print(table)
    if matrix:
        for target, pwm in resolved:
            label = f"{target.tf_name} ({target.entry.source}:{target.entry.motif_id})"
            console.print(_render_pwm_matrix(f"PWM: {label}", pwm.matrix.tolist()))
            if log_odds:
                console.print(_render_pwm_matrix(f"Log-odds: {label}", pwm.log_odds().tolist()))


@app.command("logos", help="Render PWM logos for selected TFs or motif refs.")
def logos(
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
    tf: list[str] = typer.Option([], "--tf", help="TF name to include (repeatable)."),
    ref: list[str] = typer.Option([], "--ref", help="Catalog reference (<source>:<motif_id>, repeatable)."),
    set_index: int | None = typer.Option(
        None,
        "--set",
        help="Regulator set index from config (1-based).",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Limit TF resolution to a single source adapter.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Directory to write logo PNGs (defaults to <out_dir>/logos/catalog/<run>).",
    ),
    bits_mode: str | None = typer.Option(
        None,
        "--bits-mode",
        help="Logo mode: information or probability (defaults to parse.plot.bits_mode).",
    ),
    dpi: int | None = typer.Option(
        None,
        "--dpi",
        help="DPI for logo output (defaults to parse.plot.dpi).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        ensure_mpl_cache(config_path.parent / cfg.motif_store.catalog_root)
        targets, catalog = _resolve_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source_filter=source,
        )
        store = CatalogMotifStore(
            config_path.parent / cfg.motif_store.catalog_root,
            pwm_source=cfg.motif_store.pwm_source,
            site_kinds=cfg.motif_store.site_kinds,
            combine_sites=cfg.motif_store.combine_sites,
            site_window_lengths=cfg.motif_store.site_window_lengths,
            site_window_center=cfg.motif_store.site_window_center,
            pwm_window_lengths=cfg.motif_store.pwm_window_lengths,
            pwm_window_strategy=cfg.motif_store.pwm_window_strategy,
            min_sites_for_pwm=cfg.motif_store.min_sites_for_pwm,
            allow_low_sites=cfg.motif_store.allow_low_sites,
            pseudocounts=cfg.motif_store.pseudocounts,
        )
        resolved_bits_mode = bits_mode or cfg.parse.plot.bits_mode
        resolved_dpi = dpi or cfg.parse.plot.dpi
        if resolved_bits_mode not in {"information", "probability"}:
            raise typer.BadParameter("--bits-mode must be 'information' or 'probability'.")
        out_base = out_dir
        if out_base is None:
            run_name = build_run_name(
                "catalog",
                [t.tf_name for t in targets],
                set_index=set_index,
                include_stage=False,
            )
            out_base = logos_dir_for_run(
                out_root(config_path, cfg.out_dir),
                "catalog",
                run_name,
            )
        out_base.mkdir(parents=True, exist_ok=True)
        from dnadesign.cruncher.io.plots.pssm import plot_pwm

        table = Table(title="Rendered PWM logos", header_style="bold")
        table.add_column("TF")
        table.add_column("Source")
        table.add_column("Motif ID")
        table.add_column("Length")
        table.add_column("Bits")
        table.add_column("Output")
        for target in targets:
            pwm = store.get_pwm(target.ref)
            info_bits = pwm.information_bits()
            subtitle = logo_subtitle(
                pwm_source=cfg.motif_store.pwm_source,
                entry=target.entry,
                catalog=catalog,
                combine_sites=cfg.motif_store.combine_sites,
                site_kinds=cfg.motif_store.site_kinds,
            )
            stem = _safe_stem(f"{target.tf_name}_{target.entry.source}_{target.entry.motif_id}")
            out_path = out_base / f"{stem}_logo.png"
            plot_pwm(
                pwm,
                mode=resolved_bits_mode,
                out=out_path,
                dpi=resolved_dpi,
                subtitle=f"sites: {subtitle}" if cfg.motif_store.pwm_source == "sites" else subtitle,
            )
            table.add_row(
                target.tf_name,
                target.entry.source,
                target.entry.motif_id,
                str(pwm.length),
                f"{info_bits:.2f}",
                render_path(out_path, base=config_path.parent),
            )
        console.print(table)
        console.print(f"Logos saved to {render_path(out_base, base=config_path.parent)}")
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch motifs/sites before catalog logos.")
        raise typer.Exit(code=1)
