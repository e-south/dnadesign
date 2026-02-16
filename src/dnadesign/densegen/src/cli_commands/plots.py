"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/plots.py

Plotting CLI command registration.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..cli_commands.context import CliContext
from ..utils.logging_utils import install_native_stderr_filters
from ..utils.mpl_utils import ensure_mpl_cache_dir


def register_plot_commands(app: typer.Typer, *, context: CliContext) -> None:
    console = context.console
    make_table = context.make_table

    @app.command("ls-plots", help="List available plot names and descriptions.")
    def ls_plots(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        if config is not None:
            cfg_path, is_default = context.resolve_config_path(ctx, config)
            context.load_config_or_exit(
                cfg_path,
                missing_message=context.default_config_missing_message if is_default else None,
            )
        from ..viz.plot_registry import PLOT_SPECS

        table = make_table("plot", "description")
        for name, meta in PLOT_SPECS.items():
            table.add_row(name, meta["description"])
        console.print(table)

    @app.command(help="Generate plots from outputs according to YAML. Use --only to select plots.")
    def plot(
        ctx: typer.Context,
        only: Optional[str] = typer.Option(None, help="Comma-separated plot names (subset of available plots)."),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        try:
            ensure_mpl_cache_dir()
        except Exception as exc:
            console.print(f"[bold red]Matplotlib cache setup failed:[/] {exc}")
            console.print(
                "[bold]Tip[/]: DenseGen defaults to a repo-local cache at .cache/matplotlib/densegen; "
                "set MPLCONFIGDIR to override."
            )
            raise typer.Exit(code=1)
        install_native_stderr_filters(suppress_solver_messages=False)
        from ..viz.plotting import run_plots_from_config

        run_plots_from_config(loaded.root, loaded.path, only=only, source="plot", absolute=absolute)
        console.print(":bar_chart: [bold green]Plots written.[/]")
