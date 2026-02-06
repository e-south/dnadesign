"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli_setup.py

CLI configuration resolution and environment checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import typer

from .config import ConfigError, load_config, resolve_outputs_scoped_path
from .integrations.meme_suite import require_executable

DEFAULT_CONFIG_FILENAME = "config.yaml"

log = logging.getLogger(__name__)


def _input_uses_fimo(input_cfg) -> bool:
    return str(getattr(input_cfg, "type", "")).startswith("pwm_")


def ensure_fimo_available(cfg, *, console, strict: bool = True) -> None:
    if not any(_input_uses_fimo(inp) for inp in cfg.inputs):
        return
    try:
        require_executable("fimo", tool_path=None)
    except FileNotFoundError as exc:
        msg = f"FIMO is required for this config but was not found. {exc}"
        if strict:
            console.print(f"[bold red]{msg}[/]")
            raise typer.Exit(code=1)
        log.warning(msg)


def _default_config_path() -> Path:
    return Path.cwd() / DEFAULT_CONFIG_FILENAME


def resolve_config_path(
    ctx: typer.Context,
    override: Optional[Path],
    *,
    console,
    display_path: Callable[[Path, Path, bool], str],
) -> tuple[Path, bool]:
    if override is not None:
        return Path(override), False
    if ctx.obj:
        ctx_path = ctx.obj.get("config_path")
        if ctx_path is not None:
            return Path(ctx_path), False
    env_path = os.environ.get("DENSEGEN_CONFIG_PATH")
    if env_path:
        return Path(env_path), False
    default_path = _default_config_path()
    if default_path.exists():
        return default_path, True
    console.print(
        "[bold red]No config file found.[/] Pass -c/--config, set DENSEGEN_CONFIG_PATH, "
        "or run from a workspace directory with config.yaml."
    )
    raise typer.Exit(code=1)


def load_config_or_exit(
    cfg_path: Path,
    *,
    console,
    missing_message: str | None = None,
    absolute: bool = False,
    display_root: Path | None = None,
    display_path: Callable[[Path, Path, bool], str],
):
    try:
        return load_config(cfg_path)
    except FileNotFoundError:
        if missing_message:
            console.print(f"[bold red]{missing_message}[/]")
        else:
            root = display_root or Path.cwd()
            console.print(f"[bold red]Config file not found:[/] {display_path(cfg_path, root, absolute)}")
        raise typer.Exit(code=1)
    except ConfigError as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        raise typer.Exit(code=1)


def resolve_outputs_path_or_exit(
    cfg_path: Path,
    run_root: Path,
    value: str | os.PathLike,
    *,
    label: str,
    console,
) -> Path:
    try:
        return resolve_outputs_scoped_path(cfg_path, run_root, value, label=label)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1)
