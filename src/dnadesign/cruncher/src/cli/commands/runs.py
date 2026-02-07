"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/runs.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from dnadesign.cruncher.analysis.layout import (
    current_analysis_id,
    list_analysis_entries,
)
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
from dnadesign.cruncher.artifacts.layout import live_metrics_path, manifest_path, status_path
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    parse_config_and_value,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

app = typer.Typer(no_args_is_help=True, help="List, inspect, or watch past run artifacts.")
console = Console()


def _analysis_ids(run_dir: Path) -> list[str]:
    entries = list_analysis_entries(run_dir)
    return sorted({entry["id"] for entry in entries if entry.get("id")})


def _latest_analysis_id(run_dir: Path) -> str | None:
    try:
        return current_analysis_id(run_dir)
    except (FileNotFoundError, ValueError):
        return None


def _artifact_counts(artifacts: list[dict]) -> list[dict[str, str | int]]:
    counts: dict[tuple[str, str], int] = {}
    for item in artifacts:
        stage = str(item.get("stage") or "unknown")
        kind = str(item.get("type") or "unknown")
        key = (stage, kind)
        counts[key] = counts.get(key, 0) + 1
    payload = [{"stage": stage, "type": kind, "count": count} for (stage, kind), count in sorted(counts.items())]
    return payload


def _tail_lines(path: Path, *, max_lines: int, max_bytes: int = 65536) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        block = b""
        while pos > 0 and block.count(b"\n") <= max_lines:
            read_size = min(max_bytes, pos)
            pos -= read_size
            fh.seek(pos)
            block = fh.read(read_size) + block
            if pos == 0:
                break
    lines = block.splitlines()[-max_lines:]
    return [line.decode("utf-8", errors="ignore") for line in lines if line.strip()]


def _load_live_metrics(run_dir: Path, *, max_points: int) -> list[dict]:
    metrics_path = live_metrics_path(run_dir)
    if not metrics_path.exists():
        return []
    lines = _tail_lines(metrics_path, max_lines=max_points)
    payloads: list[dict] = []
    for line in lines:
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return payloads


def _sparkline(values: list[float], *, width: int = 32) -> str:
    if not values:
        return "-"
    charset = " .:-=+*#%@"
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return "-"
    if len(cleaned) > width:
        step = max(len(cleaned) / width, 1.0)
        cleaned = [cleaned[int(i * step)] for i in range(width)]
    vmin = min(cleaned)
    vmax = max(cleaned)
    if vmax == vmin:
        return charset[-1] * len(cleaned)
    bins = len(charset) - 1
    chars = []
    for value in cleaned:
        idx = int((value - vmin) / (vmax - vmin) * bins)
        idx = max(0, min(bins, idx))
        chars.append(charset[idx])
    return "".join(chars)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _run_timestamp(run: object) -> datetime:
    created_at = _parse_timestamp(getattr(run, "created_at", None))
    if created_at is not None:
        return created_at
    run_dir = getattr(run, "run_dir", None)
    if isinstance(run_dir, Path) and run_dir.exists():
        return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
    raise FileNotFoundError(
        f"Run '{getattr(run, 'name', '-')}' has no created_at and run directory is missing: {run_dir}"
    )


def _render_index_issue_table(issues: list[object], *, title: str) -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column("Run")
    table.add_column("Stage")
    table.add_column("Issue")
    table.add_column("Run directory")
    for issue in issues:
        table.add_row(
            str(getattr(issue, "run_name", "-")),
            str(getattr(issue, "stage", "-")),
            str(getattr(issue, "reason", "-")),
            str(getattr(issue, "run_dir", "-")),
        )
    return table


def _plot_live_metrics(history: list[dict], out_path: Path) -> None:
    if not history:
        return
    best_vals = [item.get("best_score") for item in history if item.get("best_score") is not None]
    current_vals = [item.get("current_score") for item in history if item.get("current_score") is not None]
    if not best_vals and not current_vals:
        return
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    if best_vals:
        ax.plot(best_vals, label="best_score", color="steelblue")
    if current_vals:
        ax.plot(current_vals, label="current_score", color="darkorange", alpha=0.7)
    ax.set_xlabel("update")
    ax.set_ylabel("score")
    ax.set_title("Live score trend")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    cfg = load_config(config_path)
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
    cfg = load_config(config_path)
    try:
        run = get_run(cfg, config_path, run_name)
    except FileNotFoundError as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: use cruncher runs list (or add --config <path>) to see available runs.")
        raise typer.Exit(code=1)
    if json_output:
        artifacts = normalize_artifacts(run.artifacts)
        analysis_ids = _analysis_ids(run.run_dir)
        latest_analysis = _latest_analysis_id(run.run_dir)
        artifact_counts = _artifact_counts(artifacts)
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
                    "analysis_ids": analysis_ids,
                    "latest_analysis": latest_analysis,
                    "artifact_counts": artifact_counts,
                },
                indent=2,
            )
        )
        return
    console.print(f"run: {run.name}")
    console.print(f"stage: {run.stage}")
    console.print(f"status: {run.status or '-'}")
    console.print(f"created_at: {run.created_at or '-'}")
    console.print(f"motif_count: {run.motif_count}")
    if run.regulator_set:
        console.print(f"regulator_set: {run.regulator_set}")
    console.print(f"pwm_source: {run.pwm_source or '-'}")
    base = config_path.parent
    console.print(f"run_dir: {render_path(run.run_dir, base=base)}")
    artifacts = normalize_artifacts(run.artifacts)
    analysis_ids = _analysis_ids(run.run_dir)
    latest_analysis = _latest_analysis_id(run.run_dir)
    if analysis_ids:
        console.print(f"analysis_ids: {', '.join(analysis_ids)}")
    if latest_analysis:
        console.print(f"latest_analysis: {latest_analysis}")
        notebook_path = run.run_dir / "notebook__run_overview.py"
        if notebook_path.exists():
            console.print(f"notebook: {render_path(notebook_path, base=base)}")
    if artifacts:
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
        counts = _artifact_counts(artifacts)
        if counts:
            count_table = Table(title="Artifact counts", header_style="bold")
            count_table.add_column("Stage")
            count_table.add_column("Type")
            count_table.add_column("Count")
            for item in counts:
                count_table.add_row(str(item["stage"]), str(item["type"]), str(item["count"]))
            console.print(count_table)


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
    cfg = load_config(config_path)
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
    cfg = load_config(config_path)
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
    best_run = None
    best_score = None
    for run in runs:
        run_status = run.status
        score = run.best_score
        status_payload = None
        if score is None or run_status is None:
            try:
                status_payload = load_run_status(run.run_dir)
            except ValueError as exc:
                console.print(f"Error reading run status: {exc}")
                raise typer.Exit(code=1)
            if status_payload is not None:
                if score is None:
                    score = status_payload.get("best_score")
                if run_status is None:
                    run_status = status_payload.get("status")
        if str(run_status or "").lower() == "failed":
            continue
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = float(score)
            best_run = run
    if best_run is None:
        console.print("No runs with best_score found.")
        console.print("Hint: run cruncher sample to generate best_score metadata.")
        raise typer.Exit(code=1)
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
    cfg = load_config(config_path)
    index_path = rebuild_run_index(cfg, config_path)
    console.print(f"Rebuilt run index → {render_path(index_path, base=config_path.parent)}")


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
    cfg = load_config(config_path)
    issues = find_invalid_run_index_entries(config_path, stage=stage, catalog_root=cfg.catalog.catalog_root)
    if not issues:
        console.print("Run index is valid.")
        return
    console.print(_render_index_issue_table(issues, title="Invalid run index entries"))
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
    cfg = load_config(config_path)
    try:
        runs = list_runs(cfg, config_path)
    except ValueError as exc:
        console.print(f"Error reading run metadata: {exc}")
        raise typer.Exit(code=1)
    if not runs:
        console.print("No runs found.")
        raise typer.Exit(code=1)

    now = datetime.now(timezone.utc)
    stale_seconds = older_than_hours * 3600
    stale_runs: list[tuple[str, Path, dict | None, str]] = []

    for run in runs:
        if run.status != "running":
            continue
        try:
            status_payload = load_run_status(run.run_dir)
        except ValueError as exc:
            console.print(f"Error reading run status: {exc}")
            raise typer.Exit(code=1)
        last_update = None
        if status_payload is not None:
            last_update = _parse_timestamp(
                status_payload.get("updated_at")
                or status_payload.get("finished_at")
                or status_payload.get("started_at")
            )
        if last_update is None:
            reason = "missing status file" if status_payload is None else "missing timestamps"
            is_stale = True
        else:
            age = now - last_update
            is_stale = age.total_seconds() >= stale_seconds
            reason = f"no updates for {age.total_seconds() / 3600:.1f}h"
        if is_stale:
            payload = status_payload or {"stage": run.stage}
            payload["run_dir"] = str(run.run_dir.resolve())
            payload.setdefault("started_at", run.created_at or now.isoformat())
            stale_runs.append((run.name, run.run_dir, payload, reason))

    if not stale_runs:
        console.print("No stale running runs found.")
        return

    if dry_run:
        table = Table(title="Stale runs (dry-run)", header_style="bold")
        table.add_column("Run")
        table.add_column("Stage")
        table.add_column("Reason")
        for run_name, _run_dir, payload, reason in stale_runs:
            table.add_row(run_name, str(payload.get("stage") or "-"), reason)
        console.print(table)
        return

    if drop:
        removed = drop_run_index_entries(
            config_path,
            [run_name for run_name, _run_dir, _payload, _reason in stale_runs],
            catalog_root=cfg.catalog.catalog_root,
        )
        console.print(f"Dropped {removed} stale run(s) from the run index.")
        return

    updated = 0
    now_iso = now.isoformat()
    for _run_name, run_dir, payload, reason in stale_runs:
        payload["status"] = "aborted"
        payload["status_message"] = f"stale ({reason})"
        payload["updated_at"] = now_iso
        payload["finished_at"] = now_iso
        status_file = status_path(run_dir)
        atomic_write_json(status_file, payload)
        update_run_index_from_status(
            config_path,
            run_dir,
            payload,
            catalog_root=cfg.catalog.catalog_root,
        )
        updated += 1
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
    cfg = load_config(config_path)
    index_issues = find_invalid_run_index_entries(config_path, stage=stage, catalog_root=cfg.catalog.catalog_root)
    if index_issues:
        console.print(_render_index_issue_table(index_issues, title="Invalid run index entries"))
        if not repair_index:
            console.print("Error: run index contains invalid run index entries.")
            console.print("Fix: run `cruncher runs repair-index --apply` and re-run prune.")
            raise typer.Exit(code=1)
        removed = drop_run_index_entries(
            config_path,
            [issue.run_name for issue in index_issues],
            catalog_root=cfg.catalog.catalog_root,
        )
        console.print(f"Removed {removed} invalid run index entr{'y' if removed == 1 else 'ies'} before prune.")
    try:
        runs = list_runs(cfg, config_path, stage=stage)
    except ValueError as exc:
        console.print(f"Error reading run metadata: {exc}")
        raise typer.Exit(code=1)
    if not runs:
        console.print(f"No runs found for stage '{stage}'.")
        return

    runs_sorted = sorted(
        runs,
        key=lambda run: (_run_timestamp(run), run.name),
        reverse=True,
    )
    keep_names = {run.name for run in runs_sorted[:keep_latest]}
    now = datetime.now(timezone.utc)

    candidates: list[tuple[object, float]] = []
    for run in runs_sorted:
        if run.name in keep_names:
            continue
        if run.status == "running":
            continue
        run_dir = run.run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Indexed run directory is missing: {run_dir}")
        if not manifest_path(run_dir).exists():
            raise FileNotFoundError(f"Indexed run is missing run_manifest.json: {run_dir}")
        age_days = (now - _run_timestamp(run)).total_seconds() / 86400.0
        if older_than_days > 0 and age_days < older_than_days:
            continue
        candidates.append((run, age_days))

    if not candidates:
        console.print("No prune candidates found for the requested policy.")
        return

    table = Table(title=f"Run prune candidates ({stage})", header_style="bold")
    table.add_column("Run")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Age (days)", justify="right")
    for run, age_days in candidates:
        table.add_row(run.name, run.status or "-", run.created_at or "-", f"{age_days:.1f}")
    console.print(table)

    if not apply:
        console.print("Dry-run only. Re-run with --apply to archive these runs.")
        return

    archive_stage_root = config_path.parent / cfg.out_dir / "_archive" / stage
    archive_stage_root.mkdir(parents=True, exist_ok=True)
    moved_names: list[str] = []
    for run, _age_days in candidates:
        src = run.run_dir
        bucket = _run_timestamp(run).strftime("%Y-%m")
        archive_root = archive_stage_root / bucket
        archive_root.mkdir(parents=True, exist_ok=True)
        dst = archive_root / run.name
        if dst.exists():
            dst = archive_root / f"{run.name}__{int(time.time())}"
        shutil.move(str(src), str(dst))
        moved_names.append(run.name)
    removed = drop_run_index_entries(config_path, moved_names, catalog_root=cfg.catalog.catalog_root)
    console.print(
        f"Archived {len(moved_names)} run(s) to {render_path(archive_stage_root, base=config_path.parent)} "
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
    cfg = load_config(config_path)
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
    if metric_points < 1:
        console.print("Error: --metric-points must be >= 1.")
        raise typer.Exit(code=1)
    if metric_width < 1:
        console.print("Error: --metric-width must be >= 1.")
        raise typer.Exit(code=1)
    if plot_every < 1:
        console.print("Error: --plot-every must be >= 1.")
        raise typer.Exit(code=1)

    plot_target = plot_path or (run.run_dir / "live_metrics.png")

    def _render(payload: dict) -> Table:
        table = Table(title=f"Run status: {run.name}", header_style="bold")
        table.add_column("Field")
        table.add_column("Value")

        def _add_row(label: str, key: str, default: str = "-") -> None:
            value = payload.get(key, default)
            if value is None:
                value = default
            table.add_row(label, str(value))

        _add_row("stage", "stage")
        _add_row("status", "status")
        _add_row("status_message", "status_message")
        _add_row("phase", "phase")
        _add_row("chain", "chain")
        _add_row("step", "step")
        _add_row("total", "total")
        _add_row("progress_pct", "progress_pct")
        if "beta" in payload:
            _add_row("beta", "beta")
        if "beta_min" in payload or "beta_max" in payload:
            _add_row("beta_min", "beta_min")
            _add_row("beta_max", "beta_max")
        if "swap_rate" in payload:
            _add_row("swap_rate", "swap_rate")
        if "current_score" in payload:
            _add_row("current_score", "current_score")
        _add_row("best_score", "best_score")
        _add_row("best_chain", "best_chain")
        _add_row("best_draw", "best_draw")
        _add_row("acceptance_rate", "acceptance_rate")
        _add_row("updated_at", "updated_at", default=payload.get("started_at", "-"))
        if metrics:
            history = _load_live_metrics(run.run_dir, max_points=metric_points)
            if history:
                best_vals = [item.get("best_score") for item in history if item.get("best_score") is not None]
                current_vals = [item.get("current_score") for item in history if item.get("current_score") is not None]
                if best_vals:
                    table.add_row("best_score_trend", _sparkline(best_vals, width=metric_width))
                if current_vals:
                    table.add_row(
                        "current_score_trend",
                        _sparkline(current_vals, width=metric_width),
                    )
                table.add_row("metric_points", str(len(history)))
            else:
                table.add_row("live_metrics", "metrics.jsonl not found")
        return table

    tick = 0
    with Live(_render(status), refresh_per_second=4, console=console) as live:
        while True:
            try:
                status = load_run_status(run.run_dir)
            except ValueError as exc:
                console.print(f"Error reading run status: {exc}")
                raise typer.Exit(code=1)
            if status is None:
                console.print("run_status.json missing.")
                break
            live.update(_render(status))
            if plot or plot_path is not None:
                if tick % plot_every == 0:
                    history = _load_live_metrics(run.run_dir, max_points=metric_points)
                    _plot_live_metrics(history, plot_target)
            if status.get("status") in {"completed", "failed"}:
                break
            time.sleep(interval)
            tick += 1
