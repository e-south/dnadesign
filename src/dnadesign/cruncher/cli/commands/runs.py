"""Run inventory command."""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.run_service import (
    get_run,
    list_runs,
    load_run_status,
    rebuild_run_index,
)

app = typer.Typer(no_args_is_help=True, help="List, inspect, or watch past run artifacts.")
console = Console()


@app.command("list", help="List run artifacts found in the results directory.")
def list_runs_cmd(
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    stage: str | None = typer.Option(None, "--stage", help="Filter by stage (parse, sample, analyze, report)."),
) -> None:
    cfg = load_config(config)
    runs = list_runs(cfg, config, stage=stage)
    if not runs:
        console.print("No runs found.")
        console.print("Hint: run cruncher sample <config> or cruncher parse <config> to create a run.")
        raise typer.Exit(code=1)
    table = Table(title="Runs", header_style="bold")
    table.add_column("Name")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Motifs")
    table.add_column("Regulator set")
    table.add_column("PWM source")
    for run in runs:
        reg_label = "-"
        if run.regulator_set:
            idx = run.regulator_set.get("index")
            tfs = run.regulator_set.get("tfs") or []
            reg_label = f"set{idx}:" + ",".join(tfs) if idx else ",".join(tfs)
        table.add_row(
            run.name,
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
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    run_name: str = typer.Argument(..., help="Run directory name (see `cruncher runs list`).", metavar="RUN"),
) -> None:
    cfg = load_config(config)
    try:
        run = get_run(cfg, config, run_name)
    except FileNotFoundError as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: use cruncher runs list <config> to see available runs.")
        raise typer.Exit(code=1)
    console.print(f"run: {run.name}")
    console.print(f"stage: {run.stage}")
    console.print(f"status: {run.status}")
    console.print(f"created_at: {run.created_at}")
    console.print(f"motif_count: {run.motif_count}")
    if run.regulator_set:
        console.print(f"regulator_set: {run.regulator_set}")
    console.print(f"pwm_source: {run.pwm_source}")
    console.print(f"run_dir: {run.run_dir}")
    if run.artifacts:
        console.print("artifacts:")
        for item in run.artifacts:
            console.print(f"  - {item}")


@app.command("rebuild-index", help="Rebuild the run index from run_manifest.json files.")
def rebuild_index_cmd(
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
) -> None:
    cfg = load_config(config)
    index_path = rebuild_run_index(cfg, config)
    console.print(f"Rebuilt run index → {index_path}")


@app.command("watch", help="Tail run_status.json for a live progress snapshot.")
def watch_run_cmd(
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    run_name: str = typer.Argument(..., help="Run directory name (see `cruncher runs list`).", metavar="RUN"),
    interval: float = typer.Option(1.0, "--interval", help="Polling interval in seconds."),
) -> None:
    cfg = load_config(config)
    try:
        run = get_run(cfg, config, run_name)
    except FileNotFoundError as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: use cruncher runs list <config> to see available runs.")
        raise typer.Exit(code=1)
    status = load_run_status(run.run_dir)
    if status is None:
        console.print(f"No run_status.json found for run '{run_name}'.")
        console.print("Hint: watch is only available for active runs writing run_status.json.")
        raise typer.Exit(code=1)

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
        return table

    with Live(_render(status), refresh_per_second=4, console=console) as live:
        while True:
            status = load_run_status(run.run_dir)
            if status is None:
                console.print("run_status.json missing.")
                break
            live.update(_render(status))
            if status.get("status") in {"completed", "failed"}:
                break
            time.sleep(interval)
