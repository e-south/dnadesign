"""Report command."""

from __future__ import annotations

from pathlib import Path

import typer
from dnadesign.cruncher.cli.config_resolver import ConfigResolutionError, parse_config_and_value
from dnadesign.cruncher.config.load import load_config
from rich.console import Console

console = Console()


def report(
    args: list[str] = typer.Argument(
        None,
        help="Run name (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    try:
        config_path, run_name = parse_config_and_value(
            args,
            config_option,
            value_label="RUN",
            command_hint="cruncher report <run_name>",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        from dnadesign.cruncher.workflows.report_workflow import run_report

        run_report(cfg, config_path, run_name)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher sample first, then cruncher report <run> (use --config if needed).")
        raise typer.Exit(code=1)
