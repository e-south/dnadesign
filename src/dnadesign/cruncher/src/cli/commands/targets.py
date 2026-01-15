"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/targets.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.campaign_service import (
    expand_campaign,
    resolve_category_targets,
)
from dnadesign.cruncher.services.target_service import (
    has_blocking_target_errors,
    list_targets,
    target_candidates,
    target_candidates_fuzzy,
    target_stats,
    target_statuses,
)
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Check target readiness and catalog candidates.")
console = Console()


def _select_targets(
    *,
    cfg,
    config_path: Path,
    category: Optional[str],
    campaign: Optional[str],
) -> tuple:
    if category and campaign:
        raise ValueError("Use either --category or --campaign, not both.")
    if category:
        tfs = resolve_category_targets(cfg=cfg, category_name=category)
        target_cfg = cfg.model_copy(update={"regulator_sets": [tfs]})
        return target_cfg, f"Category targets: {category}", False
    if campaign:
        expansion = expand_campaign(
            cfg=cfg,
            config_path=config_path,
            campaign_name=campaign,
            include_metrics=False,
        )
        target_cfg = cfg.model_copy(update={"regulator_sets": expansion.regulator_sets})
        return target_cfg, f"Campaign targets: {campaign}", False
    return cfg, "Configured targets", True


@app.command("list", help="List configured TF targets from regulator_sets.")
def list_config_targets(
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
    category: Optional[str] = typer.Option(
        None,
        "--category",
        help="List targets from a named regulator category.",
    ),
    campaign: Optional[str] = typer.Option(
        None,
        "--campaign",
        help="List targets from a named campaign (expanded regulator_sets).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        target_cfg, title, _ = _select_targets(
            cfg=cfg,
            config_path=config_path,
            category=category,
            campaign=campaign,
        )
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    table = Table(title=title, header_style="bold")
    table.add_column("Set", style="dim", justify="right")
    table.add_column("TF")
    for set_index, tf in list_targets(target_cfg):
        table.add_row(str(set_index), tf)
    console.print(table)


@app.command("status", help="Report cached PWM/site readiness for configured targets.")
def targets_status(
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
    pwm_source: Optional[str] = typer.Option(
        None,
        "--pwm-source",
        help="Override pwm_source for preview (matrix or sites).",
    ),
    site_kinds: List[str] = typer.Option(
        [],
        "--site-kind",
        help="Limit site kinds when previewing pwm_source=sites (repeatable).",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        help="Report status for targets in a named regulator category.",
    ),
    campaign: Optional[str] = typer.Option(
        None,
        "--campaign",
        help="Report status for targets in a named campaign (expanded regulator_sets).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        target_cfg, title, use_lockfile = _select_targets(
            cfg=cfg,
            config_path=config_path,
            category=category,
            campaign=campaign,
        )
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    if pwm_source is not None and pwm_source not in {"matrix", "sites"}:
        raise typer.BadParameter("--pwm-source must be 'matrix' or 'sites'.")
    resolved_pwm_source = pwm_source or cfg.motif_store.pwm_source
    if site_kinds and resolved_pwm_source != "sites":
        raise typer.BadParameter(
            "--site-kind requires pwm_source=sites (set motif_store.pwm_source or pass --pwm-source sites)."
        )
    statuses = target_statuses(
        cfg=target_cfg,
        config_path=config_path,
        pwm_source=resolved_pwm_source,
        site_kinds=site_kinds or None,
        use_lockfile=use_lockfile,
    )
    site_label = "Sites (seq/total)"
    if target_cfg.motif_store.combine_sites:
        site_label = "Sites (merged seq/total)"
    table = Table(title=title, header_style="bold")
    table.add_column("Set", style="dim", justify="right")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Organism")
    table.add_column("Matrix")
    table.add_column(site_label)
    table.add_column("Site kind")
    table.add_column("Dataset")
    table.add_column("PWM source")
    table.add_column("Status")
    for status in statuses:
        organism = "-"
        if status.organism:
            organism = status.organism.get("name") or status.organism.get("strain") or "-"
        matrix = "yes" if status.has_matrix else "no"
        if status.matrix_source:
            matrix = f"{matrix} ({status.matrix_source})"
        if status.has_sites:
            sites = f"{status.site_count}/{status.site_total}"
        else:
            sites = "no"
        label = status.status
        if status.status == "ready":
            label = f"[green]{status.status}[/green]"
        elif status.status == "warning":
            label = f"[yellow]{status.status}[/yellow]"
        else:
            label = f"[red]{status.status}[/red]"
        table.add_row(
            str(status.set_index),
            status.tf_name,
            status.source or "-",
            status.motif_id or "-",
            organism,
            matrix,
            sites,
            status.site_kind or "-",
            status.dataset_id or "-",
            status.pwm_source,
            label,
        )
    console.print(table)
    for status in statuses:
        if status.message:
            console.print(f"- {status.tf_name}: {status.message}")
    if has_blocking_target_errors(statuses):
        console.print("Hint: resolve missing data with cruncher fetch/lock or update pwm_source in config.")
        raise typer.Exit(code=1)


@app.command("candidates", help="Show catalog candidates for each configured TF.")
def targets_candidates(
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
    fuzzy: bool = typer.Option(False, "--fuzzy", help="Use fuzzy matching for catalog candidates."),
    min_score: float = typer.Option(0.6, "--min-score", help="Minimum fuzzy score (0-1)."),
    limit: int = typer.Option(10, "--limit", help="Max candidates per TF when fuzzy."),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        help="Show candidates for targets in a named regulator category.",
    ),
    campaign: Optional[str] = typer.Option(
        None,
        "--campaign",
        help="Show candidates for targets in a named campaign (expanded regulator_sets).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        target_cfg, title, _ = _select_targets(
            cfg=cfg,
            config_path=config_path,
            category=category,
            campaign=campaign,
        )
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    if fuzzy and not (0.0 <= min_score <= 1.0):
        raise typer.BadParameter("--min-score must be between 0 and 1. Hint: try 0.6.")
    if fuzzy:
        candidates = target_candidates_fuzzy(
            cfg=target_cfg,
            config_path=config_path,
            min_score=min_score,
            limit=limit,
        )
    else:
        candidates = target_candidates(cfg=target_cfg, config_path=config_path)
    table = Table(title=title, header_style="bold")
    table.add_column("Set", style="dim", justify="right")
    table.add_column("TF")
    table.add_column("Candidates")
    for item in candidates:
        if not item.candidates:
            table.add_row(str(item.set_index), item.tf_name, "[red]none[/red]")
            continue
        entries = ", ".join(f"{c.source}:{c.motif_id}" for c in item.candidates)
        table.add_row(str(item.set_index), item.tf_name, entries)
    console.print(table)


@app.command("stats", help="Show site-length and PWM length statistics for configured targets.")
def targets_stats(
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
    category: Optional[str] = typer.Option(
        None,
        "--category",
        help="Show stats for targets in a named regulator category.",
    ),
    campaign: Optional[str] = typer.Option(
        None,
        "--campaign",
        help="Show stats for targets in a named campaign (expanded regulator_sets).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        target_cfg, title, _ = _select_targets(
            cfg=cfg,
            config_path=config_path,
            category=category,
            campaign=campaign,
        )
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    stats = target_stats(cfg=target_cfg, config_path=config_path)
    if not stats:
        console.print("No catalog entries found for configured targets.")
        console.print("Hint: run cruncher fetch motifs/sites to populate the cache.")
        raise typer.Exit(code=1)
    table = Table(title=title, header_style="bold")
    table.add_column("Set", style="dim", justify="right")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Kind")
    table.add_column("Matrix len")
    table.add_column("Sites (seq/total)")
    table.add_column("Mean len")
    table.add_column("Len min/max")
    table.add_column("Len source")
    table.add_column("Dataset")
    table.add_column("Method")
    table.add_column("Genome")
    for stat in stats:
        mean_len = "-" if stat.site_length_mean is None else f"{stat.site_length_mean:.1f}"
        min_max = "-"
        if stat.site_length_min is not None and stat.site_length_max is not None:
            min_max = f"{stat.site_length_min}/{stat.site_length_max}"
        sites = f"{stat.site_count}/{stat.site_total}" if stat.site_total else "-"
        table.add_row(
            str(stat.set_index),
            stat.tf_name,
            stat.source,
            stat.motif_id,
            stat.site_kind or "-",
            str(stat.matrix_length or "-"),
            sites,
            mean_len,
            min_max,
            stat.site_length_source or "-",
            stat.dataset_id or "-",
            stat.dataset_method or "-",
            stat.reference_genome or "-",
        )
    console.print(table)
