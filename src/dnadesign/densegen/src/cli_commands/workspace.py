"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/workspace.py

Workspace scaffolding CLI command registration.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

import typer
import yaml

from ..cli_commands.context import CliContext
from ..config import LATEST_SCHEMA_VERSION, resolve_relative_path


def register_workspace_commands(
    app: typer.Typer,
    *,
    context: CliContext,
    resolve_template_dir: Callable[..., object],
    sanitize_filename: Callable[[str], str],
    collect_relative_input_paths_from_raw: Callable[..., list[str]],
) -> None:
    console = context.console

    @app.command("init", help="Stage a new workspace with config.yaml and standard subfolders.")
    def workspace_init(
        run_id: str = typer.Option(..., "--id", "-i", help="Run identifier (directory name)."),
        root: Path = typer.Option(
            Path("."),
            "--root",
            help="Workspace root directory (default: current directory).",
        ),
        template_id: Optional[str] = typer.Option(
            None,
            "--template-id",
            help="Packaged template id (use to avoid repo-root paths).",
        ),
        template: Optional[Path] = typer.Option(None, "--template", help="Template config YAML to copy."),
        copy_inputs: bool = typer.Option(False, help="Copy file-based inputs into workspace/inputs and rewrite paths."),
    ):
        run_id_clean = sanitize_filename(run_id)
        if run_id_clean != run_id:
            console.print(f"[yellow]Sanitized run id:[/] {run_id} -> {run_id_clean}")
        root_path = root.expanduser()
        if root_path.exists() and not root_path.is_dir():
            console.print(
                "[bold red]Workspace root is not a directory:[/] "
                f"{context.display_path(root_path, Path.cwd(), absolute=False)}"
            )
            raise typer.Exit(code=1)
        run_dir = (root_path / run_id_clean).resolve()
        if run_dir.exists():
            console.print(
                "[bold red]Run directory already exists:[/] "
                f"{context.display_path(run_dir, root_path.resolve(), absolute=False)}"
            )
            raise typer.Exit(code=1)

        with resolve_template_dir(template=template, template_id=template_id) as (_template_dir, template_path):
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
            (run_dir / "outputs" / "meta").mkdir(parents=True, exist_ok=True)
            (run_dir / "outputs" / "pools").mkdir(parents=True, exist_ok=True)
            (run_dir / "outputs" / "libraries").mkdir(parents=True, exist_ok=True)
            (run_dir / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
            (run_dir / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
            (run_dir / "outputs" / "report").mkdir(parents=True, exist_ok=True)

            raw = yaml.safe_load(template_path.read_text())
            if not isinstance(raw, dict):
                console.print("[bold red]Template config must be a YAML mapping.[/]")
                raise typer.Exit(code=1)

            dense = raw.setdefault("densegen", {})
            dense["schema_version"] = LATEST_SCHEMA_VERSION
            run_block = dense.get("run") or {}
            run_block["id"] = run_id_clean
            run_block["root"] = "."
            dense["run"] = run_block

            output = dense.get("output") or {}
            if "parquet" in output and isinstance(output.get("parquet"), dict):
                output["parquet"]["path"] = "outputs/tables/dense_arrays.parquet"
            if "usr" in output and isinstance(output.get("usr"), dict):
                output["usr"]["root"] = "outputs/usr"
            dense["output"] = output

            logging_cfg = dense.get("logging") or {}
            logging_cfg["log_dir"] = "outputs/logs"
            dense["logging"] = logging_cfg

            if "plots" in raw and isinstance(raw.get("plots"), dict):
                raw["plots"]["out_dir"] = "outputs/plots"

            if copy_inputs:
                inputs_cfg = dense.get("inputs") or []
                for inp in inputs_cfg:
                    if not isinstance(inp, dict):
                        continue
                    if "path" in inp:
                        src = resolve_relative_path(template_path, inp["path"])
                        if not src.exists() or not src.is_file():
                            console.print(f"[bold red]Input file not found:[/] {src}")
                            raise typer.Exit(code=1)
                        dest = run_dir / "inputs" / src.name
                        if dest.exists():
                            console.print(f"[bold red]Input file already exists:[/] {dest}")
                            raise typer.Exit(code=1)
                        shutil.copy2(src, dest)
                        inp["path"] = str(Path("inputs") / src.name)
                    if "paths" in inp and isinstance(inp["paths"], list):
                        new_paths: list[str] = []
                        for path in inp["paths"]:
                            src = resolve_relative_path(template_path, path)
                            if not src.exists() or not src.is_file():
                                console.print(f"[bold red]Input file not found:[/] {src}")
                                raise typer.Exit(code=1)
                            dest = run_dir / "inputs" / src.name
                            if dest.exists():
                                console.print(f"[bold red]Input file already exists:[/] {dest}")
                                raise typer.Exit(code=1)
                            shutil.copy2(src, dest)
                            new_paths.append(str(Path("inputs") / src.name))
                        inp["paths"] = new_paths

            # Intentionally avoid copying auxiliary tools into the DenseGen workspace
            # to keep the workspace config-centric and low-cognitive-load.

            config_path = run_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
            if not copy_inputs:
                rel_paths = collect_relative_input_paths_from_raw(dense)
                if rel_paths:
                    console.print(
                        "[yellow]Workspace uses file-based inputs with relative paths.[/]"
                        " They will resolve relative to the new workspace."
                    )
                    for rel_path in rel_paths[:6]:
                        console.print(f"  - {rel_path}")
                    console.print("[yellow]Tip[/]: re-run with --copy-inputs or update paths in config.yaml.")
            console.print(
                ":sparkles: [bold green]Workspace staged[/]: "
                f"{context.display_path(config_path, run_dir, absolute=False)}"
            )
