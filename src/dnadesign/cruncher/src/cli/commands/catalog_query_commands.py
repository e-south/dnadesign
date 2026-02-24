"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/catalog_query_commands.py

Catalog query and inspection command registration.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from dnadesign.cruncher.app.catalog_service import get_entry, list_catalog, search_catalog
from dnadesign.cruncher.cli.catalog_execution import collect_pwm_payloads, entry_table_rows
from dnadesign.cruncher.cli.catalog_targets import _resolve_targets
from dnadesign.cruncher.cli.commands.catalog_common import console, load_config_or_exit, render_pwm_matrix
from dnadesign.cruncher.cli.config_resolver import ConfigResolutionError, parse_config_and_value, resolve_config_path
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_workspace_root


def register_catalog_query_commands(app: typer.Typer) -> None:
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
        cfg = load_config_or_exit(config_path)
        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
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
        table.add_column("Sites (cached seq/total)")
        table.add_column("Sites (matrix n)")
        table.add_column("Site kind")
        table.add_column("Dataset")
        table.add_column("Method")
        table.add_column("Mean len")
        table.add_column("Updated")
        for row in entry_table_rows(entries):
            table.add_row(
                row["tf_name"],
                row["source"],
                row["motif_id"],
                row["organism"],
                row["matrix"],
                row["sites"],
                row["matrix_sites"],
                row["site_kind"],
                row["dataset_id"],
                row["dataset_method"],
                row["mean_len"],
                row["updated"],
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
        cfg = load_config_or_exit(config_path)
        if fuzzy and regex:
            raise typer.BadParameter(
                "--fuzzy and --regex are mutually exclusive. Hint: use --fuzzy for approximate matches."
            )
        if not (0.0 <= min_score <= 1.0):
            raise typer.BadParameter("--min-score must be between 0 and 1. Hint: try 0.6.")
        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
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
        table.add_column("Sites (cached seq/total)")
        table.add_column("Sites (matrix n)")
        table.add_column("Site kind")
        table.add_column("Dataset")
        table.add_column("Method")
        table.add_column("Mean len")
        for row in entry_table_rows(entries):
            table.add_row(
                row["tf_name"],
                row["source"],
                row["motif_id"],
                row["organism"],
                row["matrix"],
                row["sites"],
                row["matrix_sites"],
                row["site_kind"],
                row["dataset_id"],
                row["dataset_method"],
                row["mean_len"],
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
        cfg = load_config_or_exit(config_path)
        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
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
        table.add_column("Sites (cached seq/total)")
        table.add_column("Sites (matrix n)")
        table.add_column("Site kind")
        table.add_column("Dataset")
        table.add_column("Method")
        table.add_column("Mean len")
        for row in entry_table_rows(entries):
            table.add_row(
                row["source"],
                row["motif_id"],
                row["organism"],
                row["matrix"],
                row["sites"],
                row["matrix_sites"],
                row["site_kind"],
                row["dataset_id"],
                row["dataset_method"],
                row["mean_len"],
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
        cfg = load_config_or_exit(config_path)
        workspace_root = resolve_workspace_root(config_path)
        if ":" not in ref:
            raise typer.BadParameter(
                "Expected <source>:<motif_id> reference. Hint: cruncher catalog show regulondb:RBM000123"
            )
        source_name, motif_id = ref.split(":", 1)
        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
        entry = get_entry(catalog_root, source=source_name, motif_id=motif_id)
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
        console.print(f"motif_path: {render_path(motif_path, base=workspace_root) if motif_path.exists() else '-'}")
        console.print(f"sites_path: {render_path(sites_path, base=workspace_root) if sites_path.exists() else '-'}")

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
        cfg = load_config_or_exit(config_path)
        if output_format not in {"table", "json"}:
            raise typer.BadParameter("--format must be 'table' or 'json'.")
        if log_odds and output_format == "table" and not matrix:
            raise typer.BadParameter("--log-odds requires --matrix for table output.")
        try:
            targets, _catalog = _resolve_targets(
                cfg=cfg,
                config_path=config_path,
                tfs=tf,
                refs=ref,
                set_index=set_index,
                source_filter=source,
            )
            catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
            payloads, resolved, rows = collect_pwm_payloads(
                cfg=cfg,
                catalog_root=catalog_root,
                targets=targets,
                output_format=output_format,
                log_odds=log_odds,
            )
        except (ValueError, FileNotFoundError) as exc:
            console.print(f"Error: {exc}")
            console.print("Hint: run cruncher fetch motifs/sites before catalog pwms.")
            raise typer.Exit(code=1)

        if output_format == "json":
            typer.echo(json.dumps(payloads, indent=2))
            return
        table = Table(title="PWM summary", header_style="bold")
        table.add_column("TF")
        table.add_column("Source")
        table.add_column("Motif ID")
        table.add_column("PWM source")
        table.add_column("Length")
        table.add_column("Window")
        table.add_column("Bits")
        table.add_column("Sites (cached seq/total)")
        table.add_column("Sites (matrix n)")
        table.add_column("Site sets")
        for row in rows:
            table.add_row(
                row["tf_name"],
                row["source"],
                row["motif_id"],
                row["pwm_source"],
                row["length"],
                row["window"],
                row["info_bits"],
                row["sites_cached"],
                row["matrix_sites"],
                row["site_sets"],
            )
        console.print(table)
        if matrix:
            for target, pwm in resolved:
                label = f"{target.tf_name} ({target.entry.source}:{target.entry.motif_id})"
                console.print(render_pwm_matrix(f"PWM: {label}", pwm.matrix.tolist()))
                if log_odds:
                    console.print(render_pwm_matrix(f"Log-odds: {label}", pwm.log_odds().tolist()))
