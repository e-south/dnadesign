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
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.numba_cache import ensure_numba_cache_dir
from rich.console import Console

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
    auto_opt: bool | None = typer.Option(
        None,
        "--auto-opt/--no-auto-opt",
        help="Run auto-optimization pilots (Gibbs + PT) and select the best candidate.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable periodic progress logging during sampling (overrides progress_every when disabled).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging (very verbose).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if verbose:
        if cfg.sample.ui.progress_every == 0:
            cfg.sample.ui.progress_every = 1000
        cfg.sample.ui.progress_bar = True
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        if cfg.sample.ui.progress_every == 0:
            cfg.sample.ui.progress_every = 1000
        cfg.sample.ui.progress_bar = True
    try:
        ensure_numba_cache_dir(config_path.parent)
        from dnadesign.cruncher.app.sample_workflow import run_sample

        run_sample(cfg, config_path, auto_opt_override=auto_opt)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch + lock, then cruncher sample <config>.")
        raise typer.Exit(code=1)
