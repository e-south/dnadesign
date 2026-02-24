"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/sample.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.numba_cache import ensure_numba_cache_dir
from dnadesign.cruncher.utils.paths import resolve_workspace_root, workspace_state_root

console = Console()


def sample(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable periodic progress logging during sampling.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging (very verbose).",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Render progress bars during sampling.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace existing run output directory before sampling.",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    try:
        cfg = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    progress_bar = bool(progress)
    progress_every = 1000 if (progress_bar and verbose) else 0
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        if progress_bar:
            progress_every = 1000
    try:
        workspace_root = resolve_workspace_root(config_path)
        cache_dir = workspace_state_root(config_path) / "numba_cache"
        ensure_numba_cache_dir(workspace_root, cache_dir=cache_dir)
        from dnadesign.cruncher.app.sample_workflow import run_sample

        run_sample(
            cfg,
            config_path,
            force_overwrite=force_overwrite,
            progress_bar=progress_bar,
            progress_every=progress_every,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch + lock, then cruncher sample <config>.")
        raise typer.Exit(code=1)
