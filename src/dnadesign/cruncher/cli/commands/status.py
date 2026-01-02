"""Status command."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.ingest.registry import default_registry
from dnadesign.cruncher.services.catalog_service import catalog_stats
from dnadesign.cruncher.services.run_service import list_runs
from dnadesign.cruncher.services.target_service import has_blocking_target_errors, target_statuses

console = Console()


def status(
    config: Path | None = typer.Argument(None, help="Path to cruncher config.yaml (required).", metavar="CONFIG"),
    runs_limit: int = typer.Option(5, "--runs", help="Number of recent runs to show."),
) -> None:
    if runs_limit < 1:
        raise typer.BadParameter("--runs must be >= 1.")
    if config is None:
        console.print("Missing CONFIG. Example: cruncher status path/to/config.yaml")
        raise typer.Exit(code=1)
    cfg = load_config(config)
    catalog_root = config.parent / cfg.motif_store.catalog_root
    lock_path = catalog_root / "locks" / f"{config.stem}.lock.json"
    sources = default_registry().list_sources()
    source_ids = ", ".join(spec.source_id for spec in sources) or "-"

    config_table = Table(title="Configuration", header_style="bold")
    config_table.add_column("Setting")
    config_table.add_column("Value")
    config_table.add_row("config", str(config))
    config_table.add_row("catalog_root", str(catalog_root))
    config_table.add_row("out_dir", str(config.parent / cfg.out_dir))
    config_table.add_row("pwm_source", cfg.motif_store.pwm_source)
    config_table.add_row("sources", source_ids)
    config_table.add_row("lockfile", "present" if lock_path.exists() else "missing")
    console.print(config_table)

    stats = catalog_stats(catalog_root)
    cache_table = Table(title="Cache", header_style="bold")
    cache_table.add_column("Metric")
    cache_table.add_column("Value")
    cache_table.add_row("entries", str(stats["entries"]))
    cache_table.add_row("motifs", str(stats["motifs"]))
    cache_table.add_row("site_sets", str(stats["site_sets"]))
    console.print(cache_table)

    statuses = target_statuses(cfg=cfg, config_path=config)
    if statuses:
        counts = Counter(status.status for status in statuses)
        blocking = sum(
            count
            for status, count in counts.items()
            if status
            in {
                "missing-lock",
                "missing-catalog",
                "missing-matrix",
                "missing-matrix-file",
                "missing-sites",
                "missing-sites-file",
                "insufficient-sites",
                "needs-window",
                "mismatch-lock",
            }
        )
        targets_table = Table(title="Targets", header_style="bold")
        targets_table.add_column("Total", justify="right")
        targets_table.add_column("Ready", justify="right")
        targets_table.add_column("Warning", justify="right")
        targets_table.add_column("Blocking", justify="right")
        targets_table.add_row(
            str(len(statuses)),
            str(counts.get("ready", 0)),
            str(counts.get("warning", 0)),
            str(blocking),
        )
        console.print(targets_table)
        if has_blocking_target_errors(statuses):
            console.print("Hint: run `cruncher targets status` for details.")
    else:
        console.print("No configured targets found in regulator_sets.")

    runs = list_runs(cfg, config_path=config)
    if not runs:
        console.print("No runs found. Use `cruncher parse` or `cruncher sample` to create one.")
        return
    stage_counts = Counter(run.stage for run in runs)
    stage_summary = ", ".join(f"{stage}:{count}" for stage, count in sorted(stage_counts.items()))
    console.print(f"Runs total: {len(runs)} ({stage_summary})")
    run_table = Table(title="Recent runs", header_style="bold")
    run_table.add_column("Run")
    run_table.add_column("Stage")
    run_table.add_column("Status")
    run_table.add_column("Created")
    for run in runs[: max(runs_limit, 1)]:
        run_table.add_row(run.name, run.stage, run.status or "-", run.created_at or "-")
    console.print(run_table)
