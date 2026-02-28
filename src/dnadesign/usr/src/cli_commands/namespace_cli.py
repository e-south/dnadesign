"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/namespace_cli.py

Typer registration helpers for USR namespace registry commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import typer


def register_namespace_commands(
    namespace_app: typer.Typer,
    *,
    ctx_args_builder: Callable[..., object],
    cmd_namespace_list: Callable[[object], None],
    cmd_namespace_show: Callable[[object], None],
    cmd_namespace_register: Callable[[object], None],
) -> None:
    @namespace_app.command("list")
    def cli_namespace_list(ctx: typer.Context) -> None:
        cmd_namespace_list(ctx_args_builder(ctx))

    @namespace_app.command("show")
    def cli_namespace_show(ctx: typer.Context, name: str = typer.Argument(...)) -> None:
        cmd_namespace_show(ctx_args_builder(ctx, name=name))

    @namespace_app.command("register")
    def cli_namespace_register(
        ctx: typer.Context,
        namespace: str = typer.Argument(...),
        columns: str = typer.Option(..., "--columns", help="Comma-separated name:type list"),
        owner: str = typer.Option("", "--owner"),
        description: str = typer.Option("", "--description"),
        overwrite: bool = typer.Option(False, "--overwrite"),
    ) -> None:
        cmd_namespace_register(
            ctx_args_builder(
                ctx,
                namespace=namespace,
                columns=columns,
                owner=owner or None,
                description=description or None,
                overwrite=overwrite,
            )
        )
