"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/runs.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from dnadesign.cruncher.app.run_service import (
    drop_run_index_entries,
    find_invalid_run_index_entries,
    get_run,
    list_runs,
    load_run_status,
    rebuild_run_index,
    update_run_index_from_status,
)
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.artifacts.entries import normalize_artifacts
from dnadesign.cruncher.artifacts.layout import status_path
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    parse_config_and_value,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.cli.runs_execution import (
    analysis_ids,
    artifact_counts,
    collect_stale_runs,
    latest_analysis_id,
    load_live_metrics,
    plot_live_metrics,
    prune_candidates_table,
    render_index_issue_table,
    render_watch_table,
    run_timestamp,
    select_best_run,
    select_prune_candidates,
    stale_runs_table,
    validate_watch_options,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_workspace_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

app = typer.Typer(no_args_is_help=True, help="List, inspect, or watch past run artifacts.")
console = Console()


def _load_config_or_exit(config_path: Path):
    try:
        return load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


def _show_run_json_payload(run: object) -> dict[str, object]:
    artifacts = normalize_artifacts(run.artifacts)
    run_analysis_ids = analysis_ids(run.run_dir)
    latest_analysis = latest_analysis_id(run.run_dir)
    artifact_count_rows = artifact_counts(artifacts)
    return {
        "name": run.name,
        "stage": run.stage,
        "status": run.status,
        "created_at": run.created_at,
        "motif_count": run.motif_count,
        "pwm_source": run.pwm_source,
        "regulator_set": run.regulator_set,
        "run_dir": str(run.run_dir),
        "artifacts": run.artifacts,
        "analysis_ids": run_analysis_ids,
        "latest_analysis": latest_analysis,
        "artifact_counts": artifact_count_rows,
    }


def _print_run_show_text(*, run: object, base: Path) -> None:
    console.print(f"run: {run.name}")
    console.print(f"stage: {run.stage}")
    console.print(f"status: {run.status or '-'}")
    console.print(f"created_at: {run.created_at or '-'}")
    console.print(f"motif_count: {run.motif_count}")
    if run.regulator_set:
        console.print(f"regulator_set: {run.regulator_set}")
    console.print(f"pwm_source: {run.pwm_source or '-'}")
    console.print(f"run_dir: {render_path(run.run_dir, base=base)}")
    artifacts = normalize_artifacts(run.artifacts)
    run_analysis_ids = analysis_ids(run.run_dir)
    latest_analysis = latest_analysis_id(run.run_dir)
    if run_analysis_ids:
        console.print(f"analysis_ids: {', '.join(run_analysis_ids)}")
    if latest_analysis:
        console.print(f"latest_analysis: {latest_analysis}")
        notebook_path = run.run_dir / "notebook__run_overview.py"
        if notebook_path.exists():
            console.print(f"notebook: {render_path(notebook_path, base=base)}")
    if not artifacts:
        return
    console.print("artifacts:")
    table = Table(header_style="bold")
    table.add_column("Stage")
    table.add_column("Type")
    table.add_column("Label")
    table.add_column("Path")
    for item in artifacts:
        path_val = item.get("path")
        rendered_path = "-" if path_val in {None, "-"} else render_path(path_val, base=base)
        table.add_row(
            str(item.get("stage") or "-"),
            str(item.get("type") or "-"),
            str(item.get("label") or "-"),
            rendered_path,
        )
    console.print(table)
    counts = artifact_counts(artifacts)
    if not counts:
        return
    count_table = Table(title="Artifact counts", header_style="bold")
    count_table.add_column("Stage")
    count_table.add_column("Type")
    count_table.add_column("Count")
    for item in counts:
        count_table.add_row(str(item["stage"]), str(item["type"]), str(item["count"]))
    console.print(count_table)


def _mark_stale_runs_aborted(
    *,
    config_path: Path,
    catalog_root: object,
    stale_runs: list[object],
    now_iso: str,
) -> int:
    updated = 0
    for stale_run in stale_runs:
        payload = stale_run.payload
        payload["status"] = "aborted"
        payload["status_message"] = f"stale ({stale_run.reason})"
        payload["updated_at"] = now_iso
        payload["finished_at"] = now_iso
        payload["run_name"] = stale_run.run_name
        status_file = status_path(stale_run.run_dir)
        atomic_write_json(status_file, payload)
        update_run_index_from_status(
            config_path,
            stale_run.run_dir,
            payload,
            run_name=stale_run.run_name,
            catalog_root=catalog_root,
        )
        updated += 1
    return updated


def _repair_prune_index_if_requested(
    *,
    config_path: Path,
    catalog_root: object,
    stage: str,
    repair_index: bool,
) -> None:
    index_issues = find_invalid_run_index_entries(config_path, stage=stage, catalog_root=catalog_root)
    if not index_issues:
        return
    console.print(render_index_issue_table(index_issues, title="Invalid run index entries"))
    if not repair_index:
        console.print("Error: run index contains invalid run index entries.")
        console.print("Fix: run `cruncher runs repair-index --apply` and re-run prune.")
        raise typer.Exit(code=1)
    removed = drop_run_index_entries(
        config_path,
        [issue.run_name for issue in index_issues],
        catalog_root=catalog_root,
    )
    console.print(f"Removed {removed} invalid run index entr{'y' if removed == 1 else 'ies'} before prune.")


def _archive_prune_candidates(
    *,
    candidates: list[object],
    archive_stage_root: Path,
) -> list[str]:
    archive_stage_root.mkdir(parents=True, exist_ok=True)
    moved_names: list[str] = []
    for candidate in candidates:
        run = candidate.run
        src = run.run_dir
        bucket = run_timestamp(run).strftime("%Y-%m")
        archive_root = archive_stage_root / bucket
        archive_root.mkdir(parents=True, exist_ok=True)
        dst = archive_root / run.name
        if dst.exists():
            dst = archive_root / f"{run.name}__{int(time.time())}"
        shutil.move(str(src), str(dst))
        moved_names.append(run.name)
    return moved_names


def _watch_status_loop(
    *,
    run: object,
    interval: float,
    metrics: bool,
    metric_points: int,
    metric_width: int,
    plot: bool,
    plot_path: Path | None,
    plot_every: int,
) -> None:
    plot_target = plot_path or (run.run_dir / "live_metrics.png")
    tick = 0
    with Live(
        render_watch_table(
            run_name=run.name,
            payload=load_run_status(run.run_dir) or {},
            run_dir=run.run_dir,
            metrics=metrics,
            metric_points=metric_points,
            metric_width=metric_width,
        ),
        refresh_per_second=4,
        console=console,
    ) as live:
        while True:
            try:
                status = load_run_status(run.run_dir)
            except ValueError as exc:
                console.print(f"Error reading run status: {exc}")
                raise typer.Exit(code=1) from exc
            if status is None:
                console.print("run_status.json missing.")
                break
            live.update(
                render_watch_table(
                    run_name=run.name,
                    payload=status,
                    run_dir=run.run_dir,
                    metrics=metrics,
                    metric_points=metric_points,
                    metric_width=metric_width,
                )
            )
            if plot or plot_path is not None:
                if tick % plot_every == 0:
                    history = load_live_metrics(run.run_dir, max_points=metric_points)
                    plot_live_metrics(history, plot_target)
            if status.get("status") in {"completed", "failed"}:
                break
            time.sleep(interval)
            tick += 1


@app.command("list", help="List run artifacts found in the results directory.")
def list_runs_cmd(
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
    stage: str | None = typer.Option(None, "--stage", help="Filter by stage (parse, sample, analyze, report)."),
    full: bool = typer.Option(False, "--full", help="Show full run names without truncation."),
    json_output: bool = typer.Option(False, "--json", help="Emit run metadata as JSON."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = _load_config_or_exit(config_path)
    try:
        runs = list_runs(cfg, config_path, stage=stage)
    except ValueError as exc:
        console.print(f"Error reading run metadata: {exc}")
        raise typer.Exit(code=1)
    if not runs:
        console.print("No runs found.")
        console.print("Hint: run cruncher sample <config> or cruncher parse <config> to create a run.")
        raise typer.Exit(code=1)
    if json_output:
        payload = []
        for run in runs:
            payload.append(
                {
                    "name": run.name,
                    "stage": run.stage,
                    "status": run.status,
                    "created_at": run.created_at,
                    "motif_count": run.motif_count,
                    "pwm_source": run.pwm_source,
                    "best_score": run.best_score,
                    "regulator_set": run.regulator_set,
                    "run_dir": str(run.run_dir),
                    "artifacts": run.artifacts,
                }
            )
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title="Runs", header_style="bold")
    table.add_column("Name", overflow="fold" if full else "ellipsis")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Motifs")
    table.add_column("Regulator set")
    table.add_column("PWM source")
    include_index = len(cfg.regulator_sets) > 1
    for run in runs:
        reg_label = "-"
        if run.regulator_set:
            idx = run.regulator_set.get("index")
            tfs = run.regulator_set.get("tfs") or []
            if idx and include_index:
                reg_label = f"set{idx}:" + ",".join(tfs)
            else:
                reg_label = ",".join(tfs)
        name = run.name
        table.add_row(
            name,
            run.stage,
            run.status or "-",
            run.created_at or "-",
            str(run.motif_count),
            reg_label,
            run.pwm_source or "-",
        )
    console.print(table)


@app.command("show", help="Show metadata and artifacts for a specific run.")
def show_run_cmd(
    args: list[str] = typer.Argument(
        None,
        help="Run name or run directory path (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit run metadata as JSON."),
) -> None:
    try:
        config_path, run_name = parse_config_and_value(
            args,
            config_option,
            value_label="RUN",
            command_hint="cruncher runs show <run_name|run_dir>",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = _load_config_or_exit(config_path)
    workspace_root = resolve_workspace_root(config_path)
    try:
        run = get_run(cfg, config_path, run_name)
    except FileNotFoundError as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: use cruncher runs list (or add --config <path>) to see available runs.")
        raise typer.Exit(code=1)
    if json_output:
        typer.echo(json.dumps(_show_run_json_payload(run), indent=2))
        return
    _print_run_show_text(run=run, base=workspace_root)


@app.command("latest", help="Print the most recent run name (optionally filtered by stage).")
def latest_run_cmd(
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
    stage: str | None = typer.Option(None, "--stage", help="Filter by stage (parse, sample, analyze, report)."),
    set_index: int | None = typer.Option(None, "--set-index", help="Filter by regulator set index."),
    json_output: bool = typer.Option(False, "--json", help="Emit run metadata as JSON."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = _load_config_or_exit(config_path)
    try:
        runs = list_runs(cfg, config_path, stage=stage)
    except ValueError as exc:
        console.print(f"Error reading run metadata: {exc}")
        raise typer.Exit(code=1)
    if set_index is not None:
        runs = [run for run in runs if (run.regulator_set or {}).get("index") == set_index]
    if not runs:
        console.print("No runs found.")
        console.print("Hint: run cruncher sample <config> or cruncher parse <config> to create a run.")
        raise typer.Exit(code=1)
    run = runs[0]
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "name": run.name,
                    "stage": run.stage,
                    "status": run.status,
                    "created_at": run.created_at,
                    "motif_count": run.motif_count,
                    "pwm_source": run.pwm_source,
                    "regulator_set": run.regulator_set,
                    "run_dir": str(run.run_dir),
                    "artifacts": run.artifacts,
                },
                indent=2,
            )
        )
        return
    console.print(run.name)


@app.command("best", help="Print the run name with the highest best_score (default stage=sample).")
def best_run_cmd(
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
    stage: str = typer.Option("sample", "--stage", help="Filter by stage (parse, sample, analyze, report)."),
    set_index: int | None = typer.Option(None, "--set-index", help="Filter by regulator set index."),
    json_output: bool = typer.Option(False, "--json", help="Emit run metadata as JSON."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = _load_config_or_exit(config_path)
    try:
        runs = list_runs(cfg, config_path, stage=stage)
    except ValueError as exc:
        console.print(f"Error reading run metadata: {exc}")
        raise typer.Exit(code=1)
    if set_index is not None:
        runs = [run for run in runs if (run.regulator_set or {}).get("index") == set_index]
    if not runs:
        console.print("No runs found.")
        console.print("Hint: run cruncher sample <config> or cruncher parse <config> to create a run.")
        raise typer.Exit(code=1)
    try:
        selected = select_best_run(runs, load_status=load_run_status)
    except ValueError as exc:
        console.print(f"Error reading run status: {exc}")
        raise typer.Exit(code=1)
    if selected is None:
        console.print("No runs with best_score found.")
        console.print("Hint: run cruncher sample to generate best_score metadata.")
        raise typer.Exit(code=1)
    best_run, best_score = selected
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "name": best_run.name,
                    "stage": best_run.stage,
                    "status": best_run.status,
                    "created_at": best_run.created_at,
                    "motif_count": best_run.motif_count,
                    "pwm_source": best_run.pwm_source,
                    "best_score": best_score,
                    "regulator_set": best_run.regulator_set,
                    "run_dir": str(best_run.run_dir),
                    "artifacts": best_run.artifacts,
                },
                indent=2,
            )
        )
        return
    console.print(best_run.name)


@app.command("rebuild-index", help="Rebuild the run index from run_manifest.json files.")
def rebuild_index_cmd(
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
    cfg = _load_config_or_exit(config_path)
    index_path = rebuild_run_index(cfg, config_path)
    console.print(f"Rebuilt run index → {render_path(index_path, base=resolve_workspace_root(config_path))}")


@app.command("repair-index", help="Drop invalid run index entries (missing run directory or manifest).")
def repair_index_cmd(
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
    stage: str | None = typer.Option(None, "--stage", help="Optional stage filter when repairing index entries."),
    apply: bool = typer.Option(False, "--apply", help="Remove invalid entries from the run index."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = _load_config_or_exit(config_path)
    issues = find_invalid_run_index_entries(config_path, stage=stage, catalog_root=cfg.catalog.catalog_root)
    if not issues:
        console.print("Run index is valid.")
        return
    console.print(render_index_issue_table(issues, title="Invalid run index entries"))
    if not apply:
        console.print("Dry-run only. Re-run with --apply to remove these index entries.")
        raise typer.Exit(code=1)
    removed = drop_run_index_entries(
        config_path,
        [issue.run_name for issue in issues],
        catalog_root=cfg.catalog.catalog_root,
    )
    console.print(f"Removed {removed} invalid run index entr{'y' if removed == 1 else 'ies'}.")


@app.command("clean", help="Mark stale running runs as aborted (or drop from index).")
def clean_runs_cmd(
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
    stale: bool = typer.Option(False, "--stale", help="Mark stale running runs as aborted."),
    older_than_hours: float = typer.Option(
        24.0,
        "--older-than-hours",
        help="Consider runs stale if no updates within this many hours.",
    ),
    drop: bool = typer.Option(False, "--drop", help="Remove stale runs from the run index instead of marking aborted."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show stale runs without modifying anything."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    if not stale:
        console.print("Nothing to clean. Pass --stale to mark stale running runs.")
        return
    if older_than_hours < 0:
        console.print("Error: --older-than-hours must be >= 0.")
        raise typer.Exit(code=1)
    cfg = _load_config_or_exit(config_path)
    try:
        runs = list_runs(cfg, config_path)
    except ValueError as exc:
        console.print(f"Error reading run metadata: {exc}")
        raise typer.Exit(code=1)
    if not runs:
        console.print("No runs found.")
        raise typer.Exit(code=1)

    now = datetime.now(timezone.utc)
    try:
        stale_runs = collect_stale_runs(
            runs,
            older_than_hours=older_than_hours,
            now=now,
            load_status=load_run_status,
        )
    except ValueError as exc:
        console.print(f"Error reading run status: {exc}")
        raise typer.Exit(code=1)

    if not stale_runs:
        console.print("No stale running runs found.")
        return

    if dry_run:
        console.print(stale_runs_table(stale_runs, title="Stale runs (dry-run)"))
        return

    if drop:
        removed = drop_run_index_entries(
            config_path,
            [stale_run.run_name for stale_run in stale_runs],
            catalog_root=cfg.catalog.catalog_root,
        )
        console.print(f"Dropped {removed} stale run(s) from the run index.")
        return

    now_iso = now.isoformat()
    updated = _mark_stale_runs_aborted(
        config_path=config_path,
        catalog_root=cfg.catalog.catalog_root,
        stale_runs=stale_runs,
        now_iso=now_iso,
    )
    console.print(f"Marked {updated} stale run(s) as aborted.")


@app.command("prune", help="Archive old runs with deterministic retention (keep latest N per stage).")
def prune_runs_cmd(
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
    stage: str = typer.Option("sample", "--stage", help="Run stage to prune (default: sample)."),
    keep_latest: int = typer.Option(20, "--keep-latest", help="Always keep at least this many most-recent runs."),
    older_than_days: float = typer.Option(
        30.0,
        "--older-than-days",
        help="Only archive runs at least this many days old (set 0 to ignore age).",
    ),
    repair_index: bool = typer.Option(
        False,
        "--repair-index",
        help="Drop invalid run index entries before pruning.",
    ),
    apply: bool = typer.Option(False, "--apply", help="Move candidate runs into stage archive."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    if keep_latest < 0:
        console.print("Error: --keep-latest must be >= 0.")
        raise typer.Exit(code=1)
    if older_than_days < 0:
        console.print("Error: --older-than-days must be >= 0.")
        raise typer.Exit(code=1)
    cfg = _load_config_or_exit(config_path)
    workspace_root = resolve_workspace_root(config_path)
    _repair_prune_index_if_requested(
        config_path=config_path,
        catalog_root=cfg.catalog.catalog_root,
        stage=stage,
        repair_index=repair_index,
    )
    try:
        runs = list_runs(cfg, config_path, stage=stage)
    except ValueError as exc:
        console.print(f"Error reading run metadata: {exc}")
        raise typer.Exit(code=1)
    if not runs:
        console.print(f"No runs found for stage '{stage}'.")
        return

    now = datetime.now(timezone.utc)
    try:
        candidates = select_prune_candidates(
            runs,
            keep_latest=keep_latest,
            older_than_days=older_than_days,
            now=now,
        )
    except FileNotFoundError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)

    if not candidates:
        console.print("No prune candidates found for the requested policy.")
        return

    console.print(prune_candidates_table(candidates, stage=stage))

    if not apply:
        console.print("Dry-run only. Re-run with --apply to archive these runs.")
        return

    archive_stage_root = workspace_root / cfg.out_dir / "_archive" / stage
    moved_names = _archive_prune_candidates(candidates=candidates, archive_stage_root=archive_stage_root)
    removed = drop_run_index_entries(config_path, moved_names, catalog_root=cfg.catalog.catalog_root)
    console.print(
        f"Archived {len(moved_names)} run(s) to {render_path(archive_stage_root, base=workspace_root)} "
        f"and removed {removed} index entr{'y' if removed == 1 else 'ies'}."
    )


@app.command("watch", help="Tail run_status.json for a live progress snapshot.")
def watch_run_cmd(
    args: list[str] = typer.Argument(
        None,
        help="Run name or run directory path (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    interval: float = typer.Option(1.0, "--interval", help="Polling interval in seconds."),
    metrics: bool = typer.Option(True, "--metrics/--no-metrics", help="Show live metrics trends if available."),
    metric_points: int = typer.Option(40, "--metric-points", help="Number of recent metric points to render."),
    metric_width: int = typer.Option(32, "--metric-width", help="Width of ASCII sparklines."),
    plot: bool = typer.Option(False, "--plot/--no-plot", help="Write a live metrics plot during watch."),
    plot_path: Path | None = typer.Option(
        None,
        "--plot-path",
        help="Optional path to write a live metrics plot (PNG) on refresh.",
    ),
    plot_every: int = typer.Option(5, "--plot-every", help="Write live plot every N refreshes."),
) -> None:
    try:
        config_path, run_name = parse_config_and_value(
            args,
            config_option,
            value_label="RUN",
            command_hint="cruncher runs watch <run_name|run_dir>",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = _load_config_or_exit(config_path)
    if plot or plot_path is not None:
        ensure_mpl_cache(resolve_catalog_root(config_path, cfg.catalog.catalog_root))
    try:
        run = get_run(cfg, config_path, run_name)
    except FileNotFoundError as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: use cruncher runs list (or add --config <path>) to see available runs.")
        raise typer.Exit(code=1)
    try:
        status = load_run_status(run.run_dir)
    except ValueError as exc:
        console.print(f"Error reading run status: {exc}")
        raise typer.Exit(code=1)
    if status is None:
        console.print(f"No run_status.json found for run '{run_name}'.")
        console.print("Hint: watch is only available for active runs writing run_status.json.")
        raise typer.Exit(code=1)
    try:
        validate_watch_options(metric_points=metric_points, metric_width=metric_width, plot_every=plot_every)
    except ValueError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    _watch_status_loop(
        run=run,
        interval=interval,
        metrics=metrics,
        metric_points=metric_points,
        metric_width=metric_width,
        plot=plot,
        plot_path=plot_path,
        plot_every=plot_every,
    )
