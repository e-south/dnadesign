"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/report.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.app.run_service import list_runs
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    parse_config_and_value,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.numba_cache import ensure_numba_cache_dir

console = Console()


def report(
    args: list[str] = typer.Argument(
        None,
        help="Run name or run directory path (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    latest: bool = typer.Option(False, "--latest", help="Use the latest sample run."),
) -> None:
    try:
        if latest:
            if args and len(args) > 1:
                raise ConfigResolutionError("When using --latest, provide at most a config path.")
            config_arg = Path(args[0]) if args else None
            config_path = resolve_config_path(config_option or config_arg)
            cfg = load_config(config_path)
            runs = list_runs(cfg, config_path, stage="sample")
            if not runs:
                raise FileNotFoundError("No sample runs found. Run `cruncher sample` first.")
            run_name = runs[0].name
        else:
            config_path, run_name = parse_config_and_value(
                args,
                config_option,
                value_label="RUN",
                command_hint="cruncher report <run_name|run_dir>",
            )
            cfg = load_config(config_path)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    try:
        ensure_numba_cache_dir(config_path.parent)
        from dnadesign.cruncher.app.report_workflow import run_report

        run_report(cfg, config_path, run_name)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher sample first, then cruncher report <run> (use --config if needed).")
        raise typer.Exit(code=1)
