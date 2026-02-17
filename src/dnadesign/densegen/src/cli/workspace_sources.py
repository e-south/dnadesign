"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli/workspace_sources.py

Workspace source resolution helpers for the DenseGen CLI.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
from importlib import resources
from pathlib import Path
from typing import Callable, Iterator, Optional

import typer


def list_packaged_workspace_ids() -> list[str]:
    package_root = resources.files("dnadesign.densegen")
    workspaces_dir = package_root.joinpath("workspaces")
    with resources.as_file(workspaces_dir) as resolved_root:
        ids = [
            child.name
            for child in Path(resolved_root).iterdir()
            if child.is_dir() and not child.name.startswith("_") and child.name not in {"runs", "archived"}
        ]
    return sorted(ids)


@contextlib.contextmanager
def resolve_workspace_source(
    *,
    source_config: Optional[Path],
    source_workspace: Optional[str],
    console,
    display_path: Callable[[Path, Path], str],
    default_config_filename: str,
) -> Iterator[tuple[Path, Path]]:
    if source_config and source_workspace:
        console.print("[bold red]Choose either --from-config or --from-workspace, not both.[/]")
        raise typer.Exit(code=1)

    if source_workspace:
        available_ids = list_packaged_workspace_ids()
        if source_workspace not in set(available_ids):
            available = ", ".join(available_ids) or "-"
            console.print(f"[bold red]Unknown source workspace:[/] {source_workspace}")
            console.print(f"[bold]Available packaged workspaces:[/] {available}")
            raise typer.Exit(code=1)
        package_root = resources.files("dnadesign.densegen")
        workspaces_dir = package_root.joinpath("workspaces")
        workspace_dir = workspaces_dir.joinpath(str(source_workspace))
        if not workspace_dir.exists():
            available = ", ".join(available_ids) or "-"
            console.print(f"[bold red]Unknown source workspace:[/] {source_workspace}")
            console.print(f"[bold]Available packaged workspaces:[/] {available}")
            raise typer.Exit(code=1)
        with resources.as_file(workspace_dir) as resolved:
            config_path = Path(resolved) / default_config_filename
            if not config_path.exists():
                console.print(
                    f"[bold red]Source workspace config not found:[/] "
                    f"{display_path(config_path, Path.cwd(), absolute=False)}"
                )
                raise typer.Exit(code=1)
            yield Path(resolved), config_path
        return

    if source_config is None:
        console.print("[bold red]No source provided.[/] Use --from-workspace or --from-config.")
        raise typer.Exit(code=1)

    source_path = source_config.expanduser().resolve()
    if not source_path.exists():
        console.print(f"[bold red]Source config not found:[/] {display_path(source_path, Path.cwd(), absolute=False)}")
        raise typer.Exit(code=1)
    if not source_path.is_file():
        console.print(
            f"[bold red]Source config path is not a file:[/] {display_path(source_path, Path.cwd(), absolute=False)}"
        )
        raise typer.Exit(code=1)
    yield source_path.parent, source_path
