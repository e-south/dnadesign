"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/parse.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.cli.campaign_targeting import resolve_runtime_targeting
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config

console = Console()


def parse(
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
    campaign: str | None = typer.Option(
        None,
        "--campaign",
        "-n",
        help="Campaign name to expand in-memory for this command.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace existing parse output directory before parsing.",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        cfg = resolve_runtime_targeting(
            cfg=cfg,
            config_path=config_path,
            command_name="parse",
            campaign_name=campaign,
        ).cfg
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    try:
        from dnadesign.cruncher.app.parse_workflow import run_parse

        run_parse(cfg, config_path, force_overwrite=force_overwrite)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch + lock before parse, and verify cache paths.")
        raise typer.Exit(code=1)
