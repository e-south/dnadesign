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
import json
import os
import re
import shlex
import shutil
import signal
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
NOTEBOOK_SERVER_STATE_FILENAME = ".densegen_notebook_server.json"
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
    on_process_start: Callable[[int], None] | None = None,
) -> bool:
    return notebook_runtime.run_marimo_command(
        command=command,
        env=env,
        browser_url=browser_url,
        open_timeout_seconds=open_timeout_seconds,
        on_browser_open_failure=on_browser_open_failure,
        on_process_start=on_process_start,
        subprocess_module=subprocess,
        time_module=time,
        url_is_reachable_fn=_url_is_reachable,
        open_browser_tab_fn=_open_browser_tab,
    )


def _process_is_running(pid: int) -> bool:
    return notebook_runtime.process_is_running(pid, os_module=os)


def _terminate_process_tree(pid: int) -> bool:
    return notebook_runtime.terminate_process_tree(pid, os_module=os, signal_module=signal, time_module=time)


def _notebook_server_state_path(run_root: Path) -> Path:
    return run_root / "outputs" / "notebooks" / NOTEBOOK_SERVER_STATE_FILENAME


def _read_notebook_server_state(*, run_root: Path) -> dict[str, object] | None:
    path = _notebook_server_state_path(run_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        _clear_notebook_server_state(run_root=run_root)
        return None
    if not isinstance(payload, dict):
        _clear_notebook_server_state(run_root=run_root)
        return None
    return payload


def _write_notebook_server_state(*, run_root: Path, pid: int, host: str, port: int, notebook_path: Path) -> None:
    path = _notebook_server_state_path(run_root)
    payload = {
        "pid": int(pid),
        "host": str(host),
        "port": int(port),
        "notebook_path": str(notebook_path.resolve()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _clear_notebook_server_state(*, run_root: Path) -> None:
    path = _notebook_server_state_path(run_root)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _release_workspace_notebook_port(*, run_root: Path, host: str, port: int, notebook_path: Path) -> bool:
    state = _read_notebook_server_state(run_root=run_root)
    if state is None:
        return False

    state_host = str(state.get("host") or "").strip()
    state_port_raw = state.get("port")
    state_notebook_raw = str(state.get("notebook_path") or "").strip()
    state_pid_raw = state.get("pid")
    try:
        state_port = int(state_port_raw)
        state_pid = int(state_pid_raw)
    except (TypeError, ValueError):
        _clear_notebook_server_state(run_root=run_root)
        return False

    if state_host != str(host).strip() or state_port != int(port):
        return False
    try:
        state_notebook = Path(state_notebook_raw).expanduser().resolve()
    except Exception:
        _clear_notebook_server_state(run_root=run_root)
        return False
    if state_notebook != notebook_path.resolve():
        return False

    if not _process_is_running(state_pid):
        _clear_notebook_server_state(run_root=run_root)
        return False

    terminated = _terminate_process_tree(state_pid)
    if terminated:
        _clear_notebook_server_state(run_root=run_root)
    return bool(terminated)


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
    notebook_path: Path | None = None,
) -> str:
    return render_notebook_template(
        NotebookTemplateContext(
            run_root=run_root,
            cfg_path=cfg_path,
            records_path=records_path,
            output_source=output_source,
            usr_root=usr_root,
            usr_dataset=usr_dataset,
            notebook_path=notebook_path,
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
    def _normalize_notebook_text(value: str) -> str:
        return re.sub(r"(Notebook:\s+[^\n]*?)\s+\(mtime=[^)]+\)", r"\1", str(value))

    records_source = _resolve_notebook_records_path(loaded=loaded, run_root=run_root, context=context)
    rendered_template = _render_notebook_template(
        run_root=run_root,
        cfg_path=cfg_path,
        records_path=records_source.records_path,
        output_source=records_source.source,
        usr_root=records_source.usr_root,
        usr_dataset=records_source.usr_dataset,
        notebook_path=notebook_path,
    )
    current_text = notebook_path.read_text() if notebook_path.exists() else ""
    if current_text == rendered_template:
        return False
    if _normalize_notebook_text(current_text) == _normalize_notebook_text(rendered_template):
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
                notebook_path=notebook_path,
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
        template_refreshed = False
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
                template_refreshed = bool(refreshed)
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
        launch_headless = mode == "run" and (headless or not open_browser)
        browser_url = _format_http_url(host_value, port_value, for_browser=True)
        if not _port_is_available(host_value, port_value):
            existing_server_reachable = mode == "run" and _url_is_reachable(browser_url)
            existing_notebook_filename: str | None = None
            existing_notebook_matches = False
            if existing_server_reachable:
                existing_notebook_filename = _running_notebook_filename(browser_url)
                if existing_notebook_filename:
                    try:
                        existing_notebook_matches = (
                            Path(existing_notebook_filename).expanduser().resolve() == notebook_path.resolve()
                        )
                    except Exception:
                        existing_notebook_matches = False
            force_fresh_due_refresh = bool(
                existing_server_reachable and existing_notebook_matches and template_refreshed
            )
            if existing_server_reachable and existing_notebook_matches and reuse_server and not force_fresh_due_refresh:
                context.console.print(
                    "[yellow]"
                    f"--port {port_value} is already serving this notebook "
                    f"on host {host_value}; reusing existing server.[/]"
                )
                context.console.print(f"[bold]Notebook URL[/]: {browser_url}")
                if should_open_browser and not _open_browser_tab(browser_url):
                    context.console.print(
                        "[yellow]Browser did not open automatically; open the Notebook URL manually.[/]"
                    )
                return
            reclaimed_same_port = False
            if (not reuse_server or force_fresh_due_refresh) and _release_workspace_notebook_port(
                run_root=run_root,
                host=host_value,
                port=port_value,
                notebook_path=notebook_path,
            ):
                if _port_is_available(host_value, port_value):
                    reclaimed_same_port = True
                    context.console.print(
                        "[yellow]"
                        f"stale workspace server on --port {port_value} was stopped; restarting on the same port "
                        "to keep notebook URLs stable.[/]"
                    )
                else:
                    context.console.print(
                        "[yellow]"
                        f"stale workspace server on --port {port_value} was stopped, but the port is still busy; "
                        "launching a fresh server on a free port.[/]"
                    )
            if reclaimed_same_port:
                browser_url = _format_http_url(host_value, port_value, for_browser=True)
            else:
                if existing_server_reachable:
                    if force_fresh_due_refresh:
                        context.console.print(
                            "[yellow]"
                            f"--port {port_value} is already serving this notebook on host {host_value}, "
                            "but the notebook template was refreshed; launching a fresh server on a free port "
                            "to avoid stale notebook state.[/]"
                        )
                    elif existing_notebook_matches:
                        context.console.print(
                            "[yellow]"
                            f"--port {port_value} is already serving this notebook on host {host_value}; "
                            "launching a fresh server on a free port because --no-reuse-server is active.[/]"
                        )
                    else:
                        existing_note = ""
                        if existing_notebook_filename:
                            existing_note = f" It currently serves `{existing_notebook_filename}`."
                        if reuse_server:
                            if existing_notebook_filename:
                                context.console.print(
                                    "[yellow]"
                                    f"--reuse-server requested but --port {port_value} serves a different notebook "
                                    f"on host {host_value}; launching a fresh server on a free port."
                                    f"{existing_note}[/]"
                                )
                            else:
                                context.console.print(
                                    "[yellow]"
                                    f"--reuse-server requested but --port {port_value} notebook identity could not be "
                                    f"verified on host {host_value}; launching a fresh server on a free port.[/]"
                                )
                        else:
                            context.console.print(
                                "[yellow]"
                                f"--port {port_value} is already serving a notebook on host {host_value}; "
                                "launching a fresh server on a free port. "
                                "Use --reuse-server to attach when it is serving this notebook."
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
        if mode == "run":
            context.console.print("[dim]Keep this process running; press Ctrl+C to stop the notebook server.[/]")
        env = dict(os.environ)
        env.setdefault("MARIMO_SKIP_UPDATE_CHECK", "1")
        if mode == "run" and should_open_browser and sys.platform == "darwin":
            env.setdefault("BROWSER", "open")
        started_pid: int | None = None

        def _on_process_start(pid: int) -> None:
            nonlocal started_pid
            started_pid = int(pid)
            if mode == "run":
                _write_notebook_server_state(
                    run_root=run_root,
                    pid=int(pid),
                    host=host_value,
                    port=port_value,
                    notebook_path=notebook_path,
                )

        run_kwargs = {
            "command": command,
            "env": env,
            "browser_url": None,
            "open_timeout_seconds": open_timeout_value,
            "on_browser_open_failure": None,
            "on_process_start": _on_process_start if mode == "run" else None,
        }
        try:
            try:
                _run_marimo_command(**run_kwargs)
            except TypeError as exc:
                if "on_process_start" not in str(exc):
                    raise
                run_kwargs.pop("on_process_start", None)
                _run_marimo_command(**run_kwargs)
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
        finally:
            if mode == "run" and started_pid is not None and not _process_is_running(started_pid):
                _clear_notebook_server_state(run_root=run_root)
