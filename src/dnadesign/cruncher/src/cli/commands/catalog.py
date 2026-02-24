"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/catalog.py

Catalog CLI command group entrypoint and command registration.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from dnadesign.cruncher.cli.commands.catalog_export_commands import register_catalog_export_commands
from dnadesign.cruncher.cli.commands.catalog_logo_commands import register_catalog_logo_commands
from dnadesign.cruncher.cli.commands.catalog_query_commands import register_catalog_query_commands

app = typer.Typer(no_args_is_help=True, help="Query or inspect cached motifs and binding sites.")

register_catalog_query_commands(app)
register_catalog_export_commands(app)
register_catalog_logo_commands(app)
