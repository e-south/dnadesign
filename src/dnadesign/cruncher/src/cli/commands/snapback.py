"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/snapback.py

CLI entrypoints for explicit v2 and co-design solve v3 snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from dnadesign.cruncher.cli.commands.snapback_explicit import design_cmd, solve_cmd, target_search_cmd, validate_cmd
from dnadesign.cruncher.cli.commands.snapback_released import (
    released_design_cmd,
    released_solve_cmd,
    released_target_search_cmd,
)
from dnadesign.cruncher.cli.commands.snapback_show import released_show_cmd, show_cmd
from dnadesign.cruncher.cli.commands.snapback_workspace import init_workspace_cmd

app = typer.Typer(
    no_args_is_help=True,
    help="Scaffold, validate, design, solve, and inspect single-nick snapback workflows.",
)

app.command("init-workspace", help="Scaffold a snapback workspace with v2 explicit and v3 co-design solve examples.")(
    init_workspace_cmd
)
app.command("validate", help="Validate a v2 explicit snapback spec and emit a deterministic report.")(validate_cmd)
app.command("design", help="Materialize one v2 explicit snapback design bundle.")(design_cmd)
app.command(
    "solve",
    help=(
        "Search for concrete snapback designs that satisfy a v3 co-design solve spec. "
        "Omitted boundary and size windows resolve to compact-first defaults."
    ),
)(solve_cmd)
app.command(
    "target-search",
    help=(
        "Search the nickase catalog for shortest preserved-site snapback hits at a requested geometry. "
        "This mode is target-first and does not assume an authored canonical top strand."
    ),
)(target_search_cmd)
app.command(
    "released-design",
    help="Materialize one explicit released-product snapback bundle from a two-stage precursor spec.",
)(released_design_cmd)
app.command(
    "released-target-search",
    help=(
        "Search paired nickase plus release-enzyme combinations for released-product snapback targets. "
        "Legacy defaults evaluate the exposed post-release bottom strand, while optional route-policy flags "
        "expand the search to retained active top or bottom strands."
    ),
)(released_target_search_cmd)
app.command(
    "released-solve",
    help=(
        "Search the released-product dual-enzyme catalog space, materialize the top default exposed-bottom hits, "
        "and optionally render one plot per hit."
    ),
)(released_solve_cmd)
app.command(
    "released-show",
    help="Read a released-product snapback bundle and print a path-oriented summary with drift checks.",
)(released_show_cmd)
app.command("show", help="Read a snapback design or solve workspace output root and print a path-oriented summary.")(
    show_cmd
)


__all__ = ["app"]
