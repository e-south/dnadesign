"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli/notebook.py

Workspace-scoped notebook commands for DenseGen.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import typer

from .context import CliContext
from .notebook_template import NotebookTemplateContext, render_notebook_template

DEFAULT_NOTEBOOK_FILENAME = "densegen_run_overview.py"


@dataclass(frozen=True)
class NotebookRecordsSource:
    source: str
    records_path: Path
    usr_root: Path | None = None
    usr_dataset: str | None = None


def _ensure_marimo_installed() -> None:
    if importlib.util.find_spec("marimo") is not None:
        return
    raise RuntimeError("marimo is not installed. Install with `uv sync --locked`.")


def _default_notebook_path(run_root: Path) -> Path:
    return run_root / "outputs" / "notebooks" / DEFAULT_NOTEBOOK_FILENAME


def _resolve_notebook_records_path(*, loaded, run_root: Path, context: CliContext) -> NotebookRecordsSource:
    output_cfg = loaded.root.densegen.output
    targets = list(output_cfg.targets)
    if not targets:
        raise ValueError("output.targets must contain at least one sink")
    if len(targets) > 1:
        plots_cfg = loaded.root.plots
        if plots_cfg is None or plots_cfg.source is None:
            raise ValueError("plots.source must be set when output.targets has multiple sinks")
        source = str(plots_cfg.source)
        if source not in targets:
            raise ValueError("plots.source must be one of output.targets")
    else:
        source = str(targets[0])

    if source == "parquet":
        parquet_cfg = output_cfg.parquet
        if parquet_cfg is None:
            raise ValueError("output.parquet is required when notebook source resolves to parquet")
        return NotebookRecordsSource(
            source="parquet",
            records_path=Path(
                context.resolve_outputs_path_or_exit(
                    loaded.path,
                    run_root,
                    parquet_cfg.path,
                    label="output.parquet.path",
                )
            ),
        )

    if source == "usr":
        usr_cfg = output_cfg.usr
        if usr_cfg is None:
            raise ValueError("output.usr is required when notebook source resolves to usr")
        dataset = str(usr_cfg.dataset).strip()
        if not dataset:
            raise ValueError("output.usr.dataset must be a non-empty string")
        usr_root = Path(
            context.resolve_outputs_path_or_exit(
                loaded.path,
                run_root,
                usr_cfg.root,
                label="output.usr.root",
            )
        )
        return NotebookRecordsSource(
            source="usr",
            records_path=usr_root / dataset / "records.parquet",
            usr_root=usr_root,
            usr_dataset=dataset,
        )

    raise ValueError(f"Unsupported notebook source: {source!r}")


def _render_notebook_template(
    *,
    run_root: Path,
    cfg_path: Path,
    records_path: Path,
    output_source: str,
    usr_root: Path | None,
    usr_dataset: str | None,
) -> str:
    return render_notebook_template(
        NotebookTemplateContext(
            run_root=run_root,
            cfg_path=cfg_path,
            records_path=records_path,
            output_source=output_source,
            usr_root=usr_root,
            usr_dataset=usr_dataset,
        )
    )


def register_notebook_commands(app: typer.Typer, *, context: CliContext) -> None:
    @app.command("generate", help="Generate a workspace-scoped marimo notebook for the current run.")
    def notebook_generate(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
        out: Optional[Path] = typer.Option(
            None,
            "--out",
            help="Notebook output path (default: <run_root>/outputs/notebooks/densegen_run_overview.py).",
        ),
        force: bool = typer.Option(False, "--force", help="Overwrite notebook if it already exists."),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
    ) -> None:
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
            absolute=absolute,
            display_root=Path.cwd(),
        )
        run_root = Path(context.run_root_for(loaded))
        try:
            records_source = _resolve_notebook_records_path(loaded=loaded, run_root=run_root, context=context)
        except Exception as exc:
            context.console.print(f"[bold red]Failed to resolve notebook records source:[/] {exc}")
            raise typer.Exit(code=1) from exc
        notebook_path = Path(out).expanduser().resolve() if out is not None else _default_notebook_path(run_root)
        if notebook_path.exists() and not force:
            context.console.print(f"[bold red]Notebook already exists:[/] {notebook_path}")
            context.console.print("[bold]Next step[/]: rerun with --force to overwrite.")
            raise typer.Exit(code=1)
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        notebook_path.write_text(
            _render_notebook_template(
                run_root=run_root,
                cfg_path=loaded.path,
                records_path=records_source.records_path,
                output_source=records_source.source,
                usr_root=records_source.usr_root,
                usr_dataset=records_source.usr_dataset,
            )
        )
        notebook_label = context.display_path(notebook_path, run_root, absolute=absolute)
        context.console.print(f":sparkles: [bold green]Notebook generated[/]: {notebook_label}")
        context.console.print("[bold]Next steps[/]:")
        default_notebook = _default_notebook_path(run_root).resolve()
        if notebook_path.resolve() == default_notebook:
            run_command = "dense notebook run"
        else:
            notebook_run_path = context.display_path(notebook_path, run_root, absolute=absolute)
            run_command = f"dense notebook run --path {shlex.quote(notebook_run_path)}"
        context.console.print(context.workspace_command(run_command, cfg_path=cfg_path, run_root=run_root))

    @app.command("run", help="Launch a DenseGen marimo notebook.")
    def notebook_run(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
        path: Optional[Path] = typer.Option(
            None,
            "--path",
            help="Notebook path (default: <run_root>/outputs/notebooks/densegen_run_overview.py).",
        ),
        mode: Literal["run", "edit"] = typer.Option(
            "edit",
            "--mode",
            help="Launch mode: run (read-only app) or edit (interactive editor).",
        ),
        headless: bool = typer.Option(
            False,
            "--headless",
            help="Run without opening a browser window (marimo run mode only).",
        ),
        open_browser: bool = typer.Option(
            True,
            "--open/--no-open",
            help="Open a browser tab automatically when launching run mode.",
        ),
        host: str = typer.Option("127.0.0.1", "--host", help="Host for marimo server."),
        port: int = typer.Option(2718, "--port", "-p", help="Port for marimo server."),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
    ) -> None:
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
            absolute=absolute,
            display_root=Path.cwd(),
        )
        run_root = Path(context.run_root_for(loaded))
        notebook_path = Path(path).expanduser().resolve() if path is not None else _default_notebook_path(run_root)
        if not notebook_path.exists():
            context.console.print(
                f"[bold red]No notebook found:[/] {context.display_path(notebook_path, run_root, absolute=absolute)}"
            )
            context.console.print("[bold]Next step[/]:")
            context.console.print(
                context.workspace_command("dense notebook generate", cfg_path=cfg_path, run_root=run_root)
            )
            raise typer.Exit(code=1)
        try:
            _ensure_marimo_installed()
        except RuntimeError as exc:
            context.console.print(f"[bold red]{exc}[/]")
            raise typer.Exit(code=1)
        if headless and mode != "run":
            context.console.print("[bold red]--headless is only supported with --mode run.[/]")
            raise typer.Exit(code=1)
        if not open_browser and mode != "run":
            context.console.print("[bold red]--open/--no-open is only supported with --mode run.[/]")
            raise typer.Exit(code=1)
        host_value = str(host).strip()
        if not host_value:
            context.console.print("[bold red]--host must be a non-empty value.[/]")
            raise typer.Exit(code=1)
        port_value = int(port)
        if port_value <= 0 or port_value > 65535:
            context.console.print("[bold red]--port must be within 1-65535.[/]")
            raise typer.Exit(code=1)
        context.console.print(
            f"[bold]Launching marimo ({mode})[/]: {context.display_path(notebook_path, run_root, absolute=absolute)}"
        )
        command = ["marimo", mode, str(notebook_path), "--host", host_value, "--port", str(port_value)]
        launch_headless = mode == "run" and (headless or not open_browser)
        if launch_headless:
            command.append("--headless")
        browser_url = f"http://{host_value}:{port_value}"
        if mode == "run":
            context.console.print(f"[bold]Notebook URL[/]: {browser_url}")
        env = dict(os.environ)
        env.setdefault("MARIMO_SKIP_UPDATE_CHECK", "1")
        if mode == "run" and not launch_headless and sys.platform == "darwin" and shutil.which("open"):
            env.setdefault("BROWSER", "open")
        try:
            subprocess.run(command, check=True, env=env)
        except FileNotFoundError:
            context.console.print("[bold red]marimo CLI not found on PATH.[/]")
            context.console.print(f"Try: uv run marimo {mode} " + str(notebook_path))
            raise typer.Exit(code=1)
        except subprocess.CalledProcessError as exc:
            context.console.print(f"[bold red]marimo exited with code {exc.returncode}[/]")
            if mode == "edit":
                notebook_run_path = context.display_path(notebook_path, run_root, absolute=absolute)
                rerun_command = f"dense notebook run --mode run --path {shlex.quote(notebook_run_path)}"
                context.console.print("[bold]Next step[/]:")
                context.console.print(context.workspace_command(rerun_command, cfg_path=cfg_path, run_root=run_root))
            raise typer.Exit(code=1)
