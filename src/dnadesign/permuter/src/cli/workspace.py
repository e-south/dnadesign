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

from dnadesign.permuter.src.core.paths import resolve_workspace_config_hint
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
    workspace_cfg = _load_or_exit(workspace)
    console.print(
        f"[green]✔[/green] workspace {workspace_cfg.scope_id}: "
        f"{workspace_cfg.config.scope.permute.protocol} -> {workspace_cfg.config.scope.output.dir}"
    )


@app.command("inspect", help="Print a compact workspace summary.")
def inspect(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Workspace directory or config.yaml path."),
):
    workspace_cfg = _load_or_exit(workspace)
    config = workspace_cfg.config
    table = Table(title=f"Permuter workspace: {workspace_cfg.scope_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("config", str(workspace_cfg.config_path))
    table.add_row("protocol", config.scope.permute.protocol)
    table.add_row("refs", config.scope.input.refs)
    table.add_row("outputs", config.scope.output.dir)
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
        except (OSError, ValueError) as exc:
            if strict:
                raise typer.BadParameter(str(exc)) from exc
            continue
        rows.append(
            {
                "id": cfg.scope_id,
                "path": str(path.parent),
                "protocol": cfg.config.scope.permute.protocol,
                "output": cfg.config.scope.output.dir,
            }
        )
    if as_json:
        typer.echo(json_lib.dumps(rows, indent=2, sort_keys=True))
        return
    table = Table(title="Permuter workspaces")
    table.add_column("Workspace")
    table.add_column("Protocol")
    table.add_column("Path")
    for row in rows:
        table.add_row(str(row["id"]), str(row["protocol"]), str(row["path"]))
    console.print(table)


def _load_or_exit(path: Path):
    try:
        return load_workspace(resolve_workspace_config_hint(path))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _find_or_exit(path: Path) -> list[Path]:
    try:
        return find_workspaces(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
