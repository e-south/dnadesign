"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/templates.py

Workspace template resolution helpers for the DenseGen CLI.

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

PACKAGED_TEMPLATES: dict[str, str] = {
    "demo_binding_sites_vanilla": "workspaces/demo_binding_sites_vanilla",
    "demo_meme_three_tfs": "workspaces/demo_meme_three_tfs",
}


def list_packaged_template_ids() -> list[str]:
    return sorted(PACKAGED_TEMPLATES.keys())


@contextlib.contextmanager
def resolve_template_dir(
    *,
    template: Optional[Path],
    template_id: Optional[str],
    console,
    display_path: Callable[[Path, Path], str],
    default_config_filename: str,
) -> Iterator[tuple[Path, Path]]:
    if template and template_id:
        console.print("[bold red]Choose either --template or --template-id, not both.[/]")
        raise typer.Exit(code=1)
    if template_id:
        rel_dir = PACKAGED_TEMPLATES.get(template_id)
        if not rel_dir:
            available = ", ".join(list_packaged_template_ids()) or "-"
            console.print(f"[bold red]Unknown template id:[/] {template_id}")
            console.print(f"[bold]Available template ids:[/] {available}")
            raise typer.Exit(code=1)
        package_root = resources.files("dnadesign.densegen")
        template_dir = package_root.joinpath(rel_dir)
        if not template_dir.exists():
            console.print(f"[bold red]Packaged template not found:[/] {rel_dir}")
            raise typer.Exit(code=1)
        with resources.as_file(template_dir) as resolved:
            config_path = Path(resolved) / default_config_filename
            if not config_path.exists():
                console.print(
                    f"[bold red]Template config not found:[/] {display_path(config_path, Path.cwd(), absolute=False)}"
                )
                raise typer.Exit(code=1)
            yield Path(resolved), config_path
        return
    if template is None:
        console.print("[bold red]No template provided.[/] Use --template-id or --template.")
        raise typer.Exit(code=1)
    template_path = template.expanduser().resolve()
    if not template_path.exists():
        console.print(
            f"[bold red]Template config not found:[/] {display_path(template_path, Path.cwd(), absolute=False)}"
        )
        raise typer.Exit(code=1)
    if not template_path.is_file():
        console.print(
            f"[bold red]Template path is not a file:[/] {display_path(template_path, Path.cwd(), absolute=False)}"
        )
        raise typer.Exit(code=1)
    yield template_path.parent, template_path
