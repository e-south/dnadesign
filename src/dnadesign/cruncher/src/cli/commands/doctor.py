"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/doctor.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.cli.config_resolver import ConfigResolutionError, resolve_config_path
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.integrations.meme_suite import check_meme_tools, resolve_tool_path

console = Console()


def doctor(
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
    tool: str | None = typer.Option(
        None,
        "--tool",
        help="Tool to check: auto, streme, or meme (defaults to discover.tool).",
    ),
    tool_path: Path | None = typer.Option(
        None,
        "--tool-path",
        help="Optional path to MEME Suite binary or bin directory (overrides config/MEME_BIN).",
    ),
) -> None:
    config_path = None
    cfg = None
    if config or config_option:
        try:
            config_path = resolve_config_path(config_option or config)
        except ConfigResolutionError as exc:
            console.print(str(exc))
            raise typer.Exit(code=1)
    else:
        try:
            config_path = resolve_config_path(None)
        except ConfigResolutionError:
            config_path = None

    if config_path is not None:
        cfg = load_config(config_path)
        console.print(f"Config: {render_path(config_path, base=config_path.parent)}")
    else:
        console.print("Config: - (not resolved)")

    resolved_tool = (tool or (cfg.discover.tool if cfg else "auto")).lower()
    resolved_path = resolve_tool_path(
        tool_path or (cfg.discover.tool_path if cfg else None),
        config_path=config_path,
    )
    try:
        ok, statuses = check_meme_tools(tool=resolved_tool, tool_path=resolved_path)
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)

    table = Table(title="MEME Suite check", header_style="bold")
    table.add_column("Tool")
    table.add_column("Status")
    table.add_column("Path")
    table.add_column("Version")
    table.add_column("Hint")
    base = config_path.parent if config_path is not None else None
    for status in statuses:
        path_val = status.path
        rendered_path = "-" if path_val in {None, "-"} else render_path(path_val, base=base)
        table.add_row(
            status.tool,
            status.status,
            rendered_path,
            status.version,
            status.hint,
        )
    console.print(table)

    if not ok:
        console.print(
            "Tip: if MEME Suite is installed via pixi, run `pixi run cruncher -- doctor -c <config>`, "
            "or set discover.tool_path (or MEME_BIN) to the MEME bin directory "
            "(see docs/guides/meme_suite.md)."
        )
        raise typer.Exit(code=1)
