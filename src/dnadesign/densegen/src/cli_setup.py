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


def _find_config_in_parents(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        candidate = root / DEFAULT_CONFIG_FILENAME
        if candidate.exists():
            return candidate
    return None


def _repo_root_from(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def _workspace_search_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("DENSEGEN_WORKSPACE_ROOT")
    if env_root:
        roots.append(Path(env_root))
    pixi_root = os.environ.get("PIXI_PROJECT_ROOT")
    if pixi_root:
        roots.append(Path(pixi_root))
    if not roots:
        repo_root = _repo_root_from(Path(__file__).resolve())
        if repo_root is not None:
            roots.append(repo_root)
    roots.append(Path.cwd())
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _auto_config_path() -> tuple[Path | None, list[Path]]:
    candidates: list[Path] = []
    for root in _workspace_search_roots():
        for base in (
            root / "src" / "dnadesign" / "densegen" / "workspaces",
            root / "workspaces",
        ):
            if not base.exists():
                continue
            for path in sorted(base.glob(f"*/{DEFAULT_CONFIG_FILENAME}")):
                candidates.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if len(unique) == 1:
        return unique[0], []
    return None, unique


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
    parent_path = _find_config_in_parents(Path.cwd())
    if parent_path is not None:
        return parent_path, False
    auto_path, candidates = _auto_config_path()
    if auto_path is not None:
        console.print(
            f"[bold yellow]Config not found in cwd; using[/] "
            f"{display_path(auto_path, Path.cwd(), False)} (auto-detected). "
            "Pass -c to select a different workspace."
        )
        return auto_path, False
    if candidates:
        console.print("[bold red]Multiple workspace configs found; use -c to select one.[/]")
        for path in candidates:
            console.print(f" - {display_path(path, Path.cwd(), False)}")
        raise typer.Exit(code=1)
    return default_path, True


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
