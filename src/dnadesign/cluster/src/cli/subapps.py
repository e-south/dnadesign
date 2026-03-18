"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/subapps.py

Cluster CLI sub-application builders.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..workspaces import (
    WorkspaceConfigError,
    builtin_workspaces_dir,
    init_workspace,
    list_builtin_workspaces,
    load_workspace_config,
)
from .resolution import resolve_workspace_context, runs_root_or_exit


def build_runs_app(
    *,
    console: Console,
) -> typer.Typer:
    runs_app = typer.Typer(help="Run store utilities")

    @runs_app.command("list")
    def runs_list(
        workspace: str | None = typer.Option(None, help="Workspace directory or packaged workspace id."),
        results_root: str | None = typer.Option(
            None,
            help="Standalone artifact root. Required unless --workspace is set.",
        ),
    ) -> None:
        from ..runs.index import list_runs

        workspace_ctx = resolve_workspace_context(workspace, "fit")
        df = list_runs(
            root=runs_root_or_exit(
                console=console,
                workspace_root=workspace_ctx.results_root,
                results_root=results_root,
                materialize=False,
            )
        )
        if df.empty:
            console.print("No runs recorded.")
            return
        tbl = Table(title="Recorded runs", show_lines=False, header_style="bold cyan")
        keep = [
            "kind",
            "run_slug",
            "alias",
            "created_utc",
            "source_kind",
            "x_col",
            "n_rows",
            "n_clusters",
            "method_id",
            "umap_slug",
            "analysis_path",
            "sweep_path",
        ]
        for key in keep:
            if key in df.columns:
                tbl.add_column(key)
        for _, row in df.iterrows():
            tbl.add_row(*[str(row.get(key, "")) for key in keep if key in df.columns])
        console.print(tbl)

    return runs_app


def build_workspaces_app(*, console: Console) -> typer.Typer:
    workspaces_app = typer.Typer(help="Workspace utilities")

    @workspaces_app.command("where")
    def workspaces_where(
        fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    ) -> None:
        payload = {
            "workspace_source_root": str(builtin_workspaces_dir().resolve()),
            "cwd": str(Path.cwd().resolve()),
        }
        fmt_norm = str(fmt).strip().lower()
        if fmt_norm == "json":
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            return
        if fmt_norm != "text":
            raise typer.BadParameter("format must be one of: text, json.")
        console.print(f"workspace_source_root: {payload['workspace_source_root']}")
        console.print(f"cwd: {payload['cwd']}")
        console.print("Local cluster workspaces are explicit directories; pass --workspace with a path or packaged id.")

    @workspaces_app.command("init")
    def workspaces_init(
        workspace_id: str = typer.Option(..., "--id", "-i", help="Workspace identifier (directory name)."),
        root: Path | None = typer.Option(
            None,
            "--root",
            help="Directory that will receive the new workspace. Defaults to the current working directory.",
        ),
    ) -> None:
        try:
            workspace_dir = init_workspace(workspace_id=workspace_id, root=root)
        except WorkspaceConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(
            json.dumps(
                {
                    "workspace_id": workspace_dir.name,
                    "workspace_dir": str(workspace_dir),
                    "config_path": str(workspace_dir / "config.yaml"),
                    "results_root": str(workspace_dir / "outputs" / "cluster"),
                },
                indent=2,
                sort_keys=True,
            )
        )

    @workspaces_app.command("list")
    def workspaces_list() -> None:
        for workspace_id in list_builtin_workspaces():
            typer.echo(workspace_id)

    @workspaces_app.command("show")
    def workspaces_show(workspace: str) -> None:
        try:
            config = load_workspace_config(workspace)
        except (FileNotFoundError, WorkspaceConfigError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(
            json.dumps(
                {
                    "workspace_id": config.workspace_id,
                    "workspace_dir": str(config.workspace_dir),
                    "config_path": str(config.config_path),
                    "results_root": str(config.results_root),
                    "sections": {
                        "fit": config.fit,
                        "umap": config.umap,
                        "analyze": config.analyze,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )

    return workspaces_app


def build_presets_app(*, console: Console) -> typer.Typer:
    presets_app = typer.Typer(help="Presets utilities")

    @presets_app.command("list")
    def presets_list() -> None:
        from ..presets.loader import load_all as load_presets

        pres = load_presets()
        if not pres:
            console.print("No presets found.")
            return
        for key in sorted(pres.keys()):
            typer.echo(f"{key} -> kind={pres[key].kind}")

    @presets_app.command("show")
    def presets_show(name: str) -> None:
        from ..presets.loader import load_all as load_presets

        pres = load_presets()
        if name not in pres:
            raise typer.BadParameter(f"Preset '{name}' not found.")
        typer.echo(json.dumps(pres[name].dict(), indent=2))

    return presets_app


__all__ = ["build_presets_app", "build_runs_app", "build_workspaces_app"]
