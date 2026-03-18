"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/app.py

Cluster CLI root app and callback wiring.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.traceback import install as rich_traceback

from ..util.warnings import configure as configure_warnings
from .commands import register_all
from .subapps import build_presets_app, build_runs_app, build_workspaces_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Cluster CLI — fit, UMAP, analyses, workspaces, and presets. Results live under a workspace artifact root or an explicit standalone results root.",  # noqa: E501
)
console = Console()


@app.callback(invoke_without_command=False)
def _global_opts(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Show full rich tracebacks with locals."),
) -> None:
    """Global flags and traceback configuration."""
    rich_traceback(show_locals=debug)
    ctx.obj = {"debug": debug}
    configure_warnings(verbose=debug)


register_all(app, console=console)
app.add_typer(build_runs_app(console=console), name="runs")
app.add_typer(build_workspaces_app(console=console), name="workspaces")
app.add_typer(build_presets_app(console=console), name="presets")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
