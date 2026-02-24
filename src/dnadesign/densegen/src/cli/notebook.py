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
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import typer

from . import notebook_runtime
from .context import CliContext
from .notebook_template import NotebookTemplateContext, render_notebook_template

DEFAULT_NOTEBOOK_FILENAME = "densegen_run_overview.py"
PORT_DISCOVERY_ATTEMPTS = notebook_runtime.PORT_DISCOVERY_ATTEMPTS
BROWSER_READY_TIMEOUT_SECONDS = notebook_runtime.BROWSER_READY_TIMEOUT_SECONDS
WILDCARD_BIND_HOSTS = notebook_runtime.WILDCARD_BIND_HOSTS


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


def _port_is_available(host: str, port: int) -> bool:
    return notebook_runtime.port_is_available(host, port, socket_module=socket)


def _resolve_tcp_bind_targets(host: str, port: int) -> list[tuple[int, int, int, tuple[object, ...]]]:
    return notebook_runtime.resolve_tcp_bind_targets(host, port, socket_module=socket)


def _find_available_port(host: str) -> int | None:
    return notebook_runtime.find_available_port(
        host,
        socket_module=socket,
        port_discovery_attempts=PORT_DISCOVERY_ATTEMPTS,
        port_is_available_fn=_port_is_available,
    )


def _url_is_reachable(url: str) -> bool:
    return notebook_runtime.url_is_reachable(
        url,
        urllib_request_module=urllib.request,
        urllib_error_module=urllib.error,
    )


def _running_notebook_filename(url: str) -> str | None:
    return notebook_runtime.running_notebook_filename(
        url,
        urllib_request_module=urllib.request,
        urllib_error_module=urllib.error,
    )


def _resolve_browser_host(host: str) -> str:
    return notebook_runtime.resolve_browser_host(host)


def _format_http_url(host: str, port: int, *, for_browser: bool) -> str:
    return notebook_runtime.format_http_url(host, port, for_browser=for_browser)


def _is_wsl() -> bool:
    return notebook_runtime.is_wsl(os_module=os, sys_module=sys, path_cls=Path)


def _open_browser_tab(url: str) -> bool:
    return notebook_runtime.open_browser_tab(
        url,
        is_wsl_fn=_is_wsl,
        shutil_module=shutil,
        sys_module=sys,
        os_module=os,
        subprocess_module=subprocess,
    )


def _run_marimo_command(
    *,
    command: list[str],
    env: dict[str, str],
    browser_url: str | None = None,
    open_timeout_seconds: float = BROWSER_READY_TIMEOUT_SECONDS,
    on_browser_open_failure: Callable[[str], None] | None = None,
) -> bool:
    return notebook_runtime.run_marimo_command(
        command=command,
        env=env,
        browser_url=browser_url,
        open_timeout_seconds=open_timeout_seconds,
        on_browser_open_failure=on_browser_open_failure,
        subprocess_module=subprocess,
        time_module=time,
        url_is_reachable_fn=_url_is_reachable,
        open_browser_tab_fn=_open_browser_tab,
    )


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


def _sync_default_notebook_template(
    *,
    loaded,
    run_root: Path,
    cfg_path: Path,
    notebook_path: Path,
    context: CliContext,
) -> bool:
    records_source = _resolve_notebook_records_path(loaded=loaded, run_root=run_root, context=context)
    rendered_template = _render_notebook_template(
        run_root=run_root,
        cfg_path=cfg_path,
        records_path=records_source.records_path,
        output_source=records_source.source,
        usr_root=records_source.usr_root,
        usr_dataset=records_source.usr_dataset,
    )
    current_text = notebook_path.read_text() if notebook_path.exists() else ""
    if current_text == rendered_template:
        return False
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(rendered_template)
    return True


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
            "run",
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
        reuse_server: bool = typer.Option(
            False,
            "--reuse-server/--no-reuse-server",
            help="Reuse an existing reachable notebook server on --host/--port instead of launching a fresh server.",
        ),
        open_timeout: float = typer.Option(
            BROWSER_READY_TIMEOUT_SECONDS,
            "--open-timeout",
            help="Seconds to wait for notebook readiness before manual-open warning (run mode with --open only).",
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
        if path is None:
            try:
                refreshed = _sync_default_notebook_template(
                    loaded=loaded,
                    run_root=run_root,
                    cfg_path=cfg_path,
                    notebook_path=notebook_path,
                    context=context,
                )
            except Exception as exc:
                context.console.print(f"[bold red]Failed to refresh notebook template:[/] {exc}")
                raise typer.Exit(code=1) from exc
            if refreshed:
                notebook_label = context.display_path(notebook_path, run_root, absolute=absolute)
                context.console.print(f"[yellow]Notebook template refreshed:[/] {notebook_label}")
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
        open_timeout_value = float(open_timeout)
        if open_timeout_value <= 0:
            context.console.print("[bold red]--open-timeout must be > 0 seconds.[/]")
            raise typer.Exit(code=1)
        if mode != "run" and open_timeout_value != BROWSER_READY_TIMEOUT_SECONDS:
            context.console.print("[bold red]--open-timeout is only supported with --mode run.[/]")
            raise typer.Exit(code=1)
        if mode == "run" and (headless or not open_browser) and open_timeout_value != BROWSER_READY_TIMEOUT_SECONDS:
            context.console.print("[bold red]--open-timeout requires --mode run with --open.[/]")
            raise typer.Exit(code=1)
        should_open_browser = mode == "run" and open_browser and not headless
        manual_browser_open = bool(should_open_browser)
        launch_headless = mode == "run" and (headless or not open_browser or manual_browser_open)
        browser_url = _format_http_url(host_value, port_value, for_browser=True)
        if not _port_is_available(host_value, port_value):
            existing_server_reachable = mode == "run" and _url_is_reachable(browser_url)
            existing_notebook_filename: str | None = None
            existing_notebook_matches = False
            if existing_server_reachable and not reuse_server:
                existing_notebook_filename = _running_notebook_filename(browser_url)
                if existing_notebook_filename:
                    try:
                        existing_notebook_matches = (
                            Path(existing_notebook_filename).expanduser().resolve() == notebook_path.resolve()
                        )
                    except Exception:
                        existing_notebook_matches = False
            if existing_server_reachable and (reuse_server or existing_notebook_matches):
                if existing_notebook_matches and not reuse_server:
                    context.console.print(
                        "[yellow]"
                        f"--port {port_value} is already serving this notebook "
                        f"on host {host_value}; reusing existing server.[/]"
                    )
                else:
                    context.console.print(
                        "[yellow]"
                        f"--port {port_value} is already serving a notebook "
                        f"on host {host_value}; reusing existing server.[/]"
                    )
                context.console.print(f"[bold]Notebook URL[/]: {browser_url}")
                if should_open_browser and not _open_browser_tab(browser_url):
                    context.console.print(
                        "[yellow]Browser did not open automatically; open the Notebook URL manually.[/]"
                    )
                return
            if existing_server_reachable:
                existing_note = ""
                if existing_notebook_filename:
                    existing_note = f" It currently serves `{existing_notebook_filename}`."
                context.console.print(
                    "[yellow]"
                    f"--port {port_value} is already serving a notebook on host {host_value}; "
                    "launching a fresh server on a free port. "
                    "Use --reuse-server to attach to the existing process."
                    f"{existing_note}[/]"
                )
            replacement_port = _find_available_port(host_value)
            if replacement_port is None or replacement_port <= 0:
                context.console.print(f"[bold red]No available port found on host {host_value}.[/]")
                context.console.print("[bold]Next step[/]: rerun with --port <free_port>.")
                raise typer.Exit(code=1)
            context.console.print(
                "[yellow]"
                f"--port {port_value} is already in use on host {host_value}; "
                f"switching to {replacement_port}.[/]"
            )
            port_value = int(replacement_port)
            browser_url = _format_http_url(host_value, port_value, for_browser=True)
        context.console.print(
            f"[bold]Launching marimo ({mode})[/]: {context.display_path(notebook_path, run_root, absolute=absolute)}"
        )
        command = ["marimo", mode, str(notebook_path), "--host", host_value, "--port", str(port_value)]
        if launch_headless:
            command.append("--headless")
        if mode == "run":
            context.console.print(f"[bold]Notebook URL[/]: {browser_url}")
            context.console.print("[dim]Keep this process running; press Ctrl+C to stop the notebook server.[/]")
        env = dict(os.environ)
        env.setdefault("MARIMO_SKIP_UPDATE_CHECK", "1")
        try:
            _run_marimo_command(
                command=command,
                env=env,
                browser_url=browser_url if manual_browser_open else None,
                open_timeout_seconds=open_timeout_value,
                on_browser_open_failure=(
                    lambda reason: context.console.print(
                        "[yellow]Notebook server did not become reachable in time; "
                        "open the Notebook URL manually once it is ready.[/]"
                        if reason == "notebook-not-reachable"
                        else "[yellow]Browser did not open automatically; open the Notebook URL manually.[/]"
                    )
                )
                if manual_browser_open
                else None,
            )
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
