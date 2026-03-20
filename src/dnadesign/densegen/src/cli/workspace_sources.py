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
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Callable, Iterator, Optional

import typer

PACKAGED_WORKSPACE_IDS: tuple[str, ...] = (
    "demo_tfbs_baseline",
    "demo_sampling_baseline",
    "study_constitutive_sigma_panel",
    "study_stress_ethanol_cipro",
)


def list_packaged_workspace_ids() -> list[str]:
    return [str(entry["workspace_id"]) for entry in list_packaged_workspace_inventory()]


def _workspace_inventory_entry(*, workspace_id: str, workspace_dir: Path) -> dict[str, object]:
    outputs_dir = workspace_dir / "outputs"
    output_files = 0
    latest_output_timestamp: float | None = None
    if outputs_dir.exists():
        for candidate in outputs_dir.rglob("*"):
            if not candidate.is_file():
                continue
            output_files += 1
            try:
                stat_result = candidate.stat()
            except OSError:
                continue
            if latest_output_timestamp is None or stat_result.st_mtime > latest_output_timestamp:
                latest_output_timestamp = stat_result.st_mtime
    latest_output_mtime = (
        datetime.fromtimestamp(latest_output_timestamp).astimezone().isoformat(timespec="seconds")
        if latest_output_timestamp is not None
        else None
    )
    return {
        "workspace_id": workspace_id,
        "workspace_dir": str(workspace_dir.resolve()),
        "workspace_state": "attention" if output_files else "clean",
        "output_files": output_files,
        "latest_output_mtime": latest_output_mtime,
    }


def list_packaged_workspace_inventory() -> list[dict[str, object]]:
    package_root = resources.files("dnadesign.densegen")
    workspaces_dir = package_root.joinpath("workspaces")
    with resources.as_file(workspaces_dir) as resolved_root:
        missing: list[str] = []
        available: list[dict[str, object]] = []
        for workspace_id in PACKAGED_WORKSPACE_IDS:
            cfg_path = Path(resolved_root) / workspace_id / "config.yaml"
            if not cfg_path.is_file():
                missing.append(str(workspace_id))
                continue
            available.append(_workspace_inventory_entry(workspace_id=str(workspace_id), workspace_dir=cfg_path.parent))
    if missing:
        raise RuntimeError("Packaged workspaces are missing config.yaml: " + ", ".join(missing))
    return available


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
        try:
            available_ids = list_packaged_workspace_ids()
        except RuntimeError as exc:
            console.print(f"[bold red]{exc}[/]")
            raise typer.Exit(code=1) from exc
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
