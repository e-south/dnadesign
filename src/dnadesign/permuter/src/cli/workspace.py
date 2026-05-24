"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/workspace.py

Workspace CLI commands.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json as json_lib
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.permuter.src.workspaces.loader import find_workspaces, load_workspace

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Validate and inspect Permuter workspaces.",
)
console = Console()


@app.command("validate", help="Validate a workspace config.yaml without executing runs.")
def validate(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Workspace directory or config.yaml path."),
):
    cfg = _load_or_exit(workspace)
    run_word = "run" if len(cfg.runs) == 1 else "runs"
    console.print(f"[green]✔[/green] workspace {cfg.workspace.id}: {len(cfg.runs)} {run_word}")


@app.command("inspect", help="Print a compact workspace summary.")
def inspect(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Workspace directory or config.yaml path."),
):
    cfg = _load_or_exit(workspace)
    table = Table(title=f"Permuter workspace: {cfg.workspace.id}")
    table.add_column("Run")
    table.add_column("Kind")
    table.add_column("Target")
    for run in cfg.runs:
        if run.job:
            table.add_row(run.id, "job", run.job)
        else:
            table.add_row(run.id, "protocol", str(run.protocol))
    console.print(table)


@app.command("list", help="List workspaces under a root directory.")
def list_(
    root: Path = typer.Option(Path("."), "--root", "-r", help="Directory to scan."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    strict: bool = typer.Option(False, "--strict", help="Fail if a discovered config.yaml is not a workspace."),
):
    paths = _find_or_exit(root)
    rows = []
    for path in paths:
        try:
            cfg = load_workspace(path)
        except ValueError as exc:
            if strict:
                raise typer.BadParameter(str(exc)) from exc
            continue
        rows.append({"id": cfg.workspace.id, "path": str(path.parent), "runs": len(cfg.runs)})
    if as_json:
        console.print(json_lib.dumps(rows, indent=2, sort_keys=True))
        return
    table = Table(title="Permuter workspaces")
    table.add_column("Workspace")
    table.add_column("Runs")
    table.add_column("Path")
    for row in rows:
        table.add_row(str(row["id"]), str(row["runs"]), str(row["path"]))
    console.print(table)


def _load_or_exit(path: Path):
    try:
        return load_workspace(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _find_or_exit(path: Path) -> list[Path]:
    try:
        return find_workspaces(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
